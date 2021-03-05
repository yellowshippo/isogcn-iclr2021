import argparse
from distutils.util import strtobool
import pathlib

import scipy.io as io
import scipy.sparse as sp

import femio
import numpy as np
import siml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_path',
        type=pathlib.Path,
        help='Pretrained model path')
    parser.add_argument(
        'preprocessed_data_directory',
        type=pathlib.Path,
        help='Preprocessed data directory')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-l', '--log-out-dir',
        type=pathlib.Path,
        default=None,
        help='Log output directory name')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Preprocessors.pkl file')
    parser.add_argument(
        '-w', '--write-simulation-base',
        type=pathlib.Path,
        default=None,
        help='Simulation base directory to write inferred data')
    parser.add_argument(
        '-v', '--variable-name',
        type=str,
        default=None,
        help='Variable name for inference')
    parser.add_argument(
        '-a', '--analyse-error',
        type=strtobool,
        default=1,
        help='If True, analyse MSE loss [True]')
    parser.add_argument(
        '-t', '--output-cpp-directory',
        type=pathlib.Path,
        default=None,
        help='If fed, output test data for C++ program [False]')

    args = parser.parse_args()

    if args.model_path.is_dir():
        yaml_file = args.model_path / 'settings.yml'
    else:
        yaml_file = args.model_path.parent / 'settings.yml'

    if args.write_simulation_base:
        write_simulation = True
    else:
        write_simulation = False

    if args.output_cpp_directory is not None:
        perform_postprocess = False
    else:
        perform_postprocess = True

    inferer = siml.inferer.Inferer.read_settings(yaml_file)
    results = inferer.infer(
        model=args.model_path, save=True,
        output_directory=args.out_dir,
        preprocessed_data_directory=args.preprocessed_data_directory,
        write_simulation=write_simulation,
        write_simulation_base=args.write_simulation_base,
        write_simulation_type='ucd',
        converter_parameters_pkl=args.preprocessors_pkl,
        convert_to_order1=True,
        read_simulation_type='fistr',
        write_yaml=True, perform_postprocess=perform_postprocess)

    if not args.analyse_error:
        return

    if args.variable_name is None:
        args.variable_name = list(results[0]['dict_y'].keys())[-1]
        print(f"Set variable name for error analysis: {args.variable_name}")


    print('--')
    if np.all(['loss' in r for r in results]) \
            and np.all([r['loss'] is not None for r in results]):

        error_calculator = ErrorCalculator(
            results, args.variable_name, log_out_dir=args.log_out_dir)
        error_calculator.calculate_errors()
        error_calculator.print()
        error_calculator.write()

    if args.output_cpp_directory is not None:
        cpp_data_generator = CppDataGenerator(
            inferer, results, args.output_cpp_directory)
        cpp_data_generator.write()

    return


class CppDataGenerator:

    def __init__(self, inferer, results, output_directory):
        self.inferer = inferer
        self.results = results
        self.output_directory = output_directory
        return

    def write(self):
        self.output_directory.mkdir(exist_ok=True, parents=True)
        siml.setting.write_yaml(
            self.inferer.setting, self.output_directory / 'settings.yml')
        self._write_model()

        for i_result, result in enumerate(self.results):
            self._write_resuts(i_result, result)

        print(f"Data for C++ is written in: {self.output_directory}")
        return

    def _write_resuts(self, result_count, result):
        input_names = [d['name'] for d in self.inferer.setting.trainer.inputs]
        if isinstance(input_names, dict):
            raise NotImplementedError
        output_names = [
            d['name'] for d in self.inferer.setting.trainer.outputs]
        if isinstance(output_names, dict):
            raise NotImplementedError

        output_directory = self.output_directory / f"data/{result_count}"
        output_directory.mkdir(exist_ok=True, parents=True)
        self._write_result_data(
            output_directory / 'x.dense', result['dict_x'], input_names)
        self._write_result_data(
            output_directory / 'target.dense', result['dict_x'], output_names,
            mandatory=False)
        self._write_result_data(
            output_directory / 'output.dense', result['dict_y'], output_names)
        self._write_sparse_data(output_directory, result['data_directory'])
        return

    def _write_sparse_data(self, output_directory, data_directory):
        sparse_names = self.inferer.setting.trainer.support_inputs
        for i_sparse, sparse_name in enumerate(sparse_names):
            sparse = sp.load_npz(data_directory / (sparse_name + '.npz'))
            io.mmwrite(
                output_directory / f"sparse_{i_sparse}.mtx", sparse,
                symmetry='general')
        return

    def _write_result_data(
            self, output_file_name, data_dict, names, mandatory=True):
        if not mandatory:
            if np.any([name not in data_dict for name in names]):
                print(f"Skipped writing since no data for: {output_file_name}")
                return
        self._write_dense(
            output_file_name,
            np.concatenate([data_dict[name] for name in names], axis=-1))
        return

    def _write_model(self):
        parameter_dict = dict(self.inferer.model.named_parameters())
        block_count = 0
        for block_name in self.inferer.model.sorted_graph_nodes:
            for parameter_key, parameter_value in parameter_dict.items():
                if block_name in parameter_key:
                    if 'weight' in parameter_key:
                        self._write_block(
                            block_name, parameter_key,
                            parameter_dict, block_count)
                        block_count += 1
        return

    def _write_block(
            self, block_name, weight_key, parameter_dict, block_count):
        block_setting = self.inferer.model.dict_block_setting[block_name]
        local_block_count = int(weight_key.split('.')[-2])

        output_directory = self.output_directory / f"model/{block_count}"
        output_directory.mkdir(exist_ok=True, parents=True)

        # Write block setting file
        with open(output_directory / 'setting.txt', 'w') as f:
            f.write(f"has_residual: {str(block_setting.residual).lower()}\n")
            if block_setting.type == 'iso_gcn':
                f.write('perform_spmv: true\n')
            elif 'mlp' in block_setting.type:
                f.write('perform_spmv: false\n')
            else:
                raise ValueError(
                    f"Unexpected block type: {block_setting.type}")
            activation = block_setting.activations[local_block_count]
            f.write(f"activation: {activation}\n")
            if block_setting.coeff is not None:
                f.write(f"coeff: {block_setting.coeff}\n")

        # Write parameters
        self._write_dense(
            output_directory / 'weight.dense',
            parameter_dict[weight_key].detach().numpy().T)
        bias_key = weight_key.replace('.weight', '.bias')
        self._write_dense(
            output_directory / 'bias.dense',
            parameter_dict[bias_key].detach().numpy())
        return

    def _write_dense(self, output_file_name, data):
        shape = data.shape
        if len(shape) == 1:
            row = 1
            col = data.shape[0]
        elif len(shape) == 2:
            row, col = shape
        else:
            raise ValueError(f"Unexpected shape: {shape}")
        with open(output_file_name, 'w') as f:
            f.write(f"{row}\n")
            f.write(f"{col}\n")
            np.savetxt(f, np.ravel(data))
        return


class ErrorCalculator:

    def __init__(self, results, target_variable_name, *, log_out_dir=None):
        self.target_variable_name = target_variable_name
        self.log_out_dir = log_out_dir
        self.results = results

        self._extract_variables()
        return

    def calculate_errors(self):
        self.mse, self.std_error = self._calculate_loss_stats()
        return

    def print(self):
        print(
            f"MSE of {self.target_variable_name}: {self.mse:.5e} "
            f"+/- {self.std_error:.5e}")
        return

    def write(self):
        if self.log_out_dir is None:
            print('log_out_dir is None, so no log file written')
            return

        output_directory = self.log_out_dir
        output_directory.mkdir(parents=True, exist_ok=True)
        output_file = output_directory / 'score.log'
        with open(output_file, 'w') as f:
            f.write(
                f"MSE of {self.target_variable_name}: {self.mse:.5e} "
                f"+/- {self.std_error:.5e}\n")
        print(f"Scores written in: {output_file}")
        return

    def _extract_variable(self, variable_name, allow_not_found=False):
        if variable_name in self.results[0]['dict_x']:
            variable = np.concatenate([
                r['dict_x'][variable_name] for r in self.results])
        else:
            file_paths = [
                pathlib.Path(str(r['data_directory']).replace(
                    'preprocessed', 'interim')) / f"{variable_name}.npy"
                for r in self.results]
            if np.all([p.exists() for p in file_paths]):
                variable = np.concatenate([
                    np.load(p) for p in file_paths])
            else:
                if allow_not_found:
                    return None
                else:
                    raise ValueError(f"File not found for: {variable_name}")

        return variable

    def _extract_variables(self):
        self.nodal_volume = self._extract_variable(
            'nodal_effective_volume', allow_not_found=True)
        self.has_volume = self.nodal_volume is not None
        self.target_variable = self._extract_variable(
            self.target_variable_name)

        if self.target_variable_name in self.results[0]['dict_y']:
            self.inferred_variable = np.concatenate([
                r['dict_y'][self.target_variable_name] for r in self.results])

        else:
            raise ValueError(
                f"Unexpected inferred data: {self.results[0]['dict_y']}")

        return

    def _calculate_loss_stats(self):
        return calculate_loss_stats(
            self.inferred_variable, self.target_variable)


def calculate_loss_stats(inferred_variable, target_variable):
    n = len(target_variable)
    shape = target_variable.shape
    square_error = (np.reshape(inferred_variable, shape) - target_variable)**2
    mean_square_error = np.mean(square_error)
    std_error = np.std(square_error) / np.sqrt(n)
    return mean_square_error, std_error


if __name__ == '__main__':
    main()
