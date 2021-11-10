import argparse
import pathlib

import siml

import convert_raw_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_path',
        type=pathlib.Path,
        help='Pretrained model path')
    parser.add_argument(
        'raw_data_directory',
        type=pathlib.Path,
        help='Raw data directory')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Preprocessors.pkl file')
    # parser.add_argument(
    #     '-w', '--write-simulation-base',
    #     type=pathlib.Path,
    #     default=None,
    #     help='Simulation base directory to write inferred data')
    parser.add_argument(
        '-v', '--variable-name',
        type=str,
        default='nodal_t_100',
        help='Variable name for inference')
    args = parser.parse_args()

    if args.model_path.is_dir():
        yaml_file = args.model_path / 'settings.yml'
    else:
        yaml_file = args.model_path.parent / 'settings.yml'

    # if args.write_simulation_base:
    #     write_simulation = True
    # else:
    #     write_simulation = False

    inferer = siml.inferer.Inferer.read_settings(yaml_file)
    inferer.setting.conversion.time_series = True
    inferer.infer(
        model=args.model_path, save=True,
        output_directory=args.out_dir,
        raw_data_directory=args.raw_data_directory,
        write_simulation=True,
        write_simulation_base=args.raw_data_directory,
        write_simulation_type='ucd',
        converter_parameters_pkl=args.preprocessors_pkl,
        convert_to_order1=True,
        read_simulation_type='fistr',
        conversion_function=convert_raw_data.HeatConversionFuncionCreator(
            convert_answer=False, light=True, create_elemental=False),
        write_yaml=True)


if __name__ == '__main__':
    main()
