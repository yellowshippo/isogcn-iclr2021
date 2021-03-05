"""Generate raw data."""

import argparse
from distutils.util import strtobool
from glob import glob
import multiprocessing as multi
import pathlib
import random
import subprocess

import femio
import numpy as np
import siml


RAW_DIRECTORY = pathlib.Path('data/raw')

CLSCALES = (.1, .2, .3)

MIN_FOURIER_DEGREE = 2
MAX_FOURIER_DEGREE = 10
CONDUCTIVITIES = np.arange(.5e-2, 1.5e-2 + 1e-5, .1e-2)
NL_CONDUCTIVITY = (.02, .01)

MESHING_SUCCESS_FILE = pathlib.Path('meshing_success')
GENERATION_SUCCESS_FILE = pathlib.Path('generation_success')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_cad_directories',
        type=pathlib.Path,
        nargs='+',
        help='Input CAD directories')
    parser.add_argument(
        '--output-directory',
        '-o',
        type=pathlib.Path,
        default=RAW_DIRECTORY,
        help='Output base direcoty')
    parser.add_argument(
        '--clscales',
        '-s',
        type=float,
        nargs='+',
        default=CLSCALES,
        help='Mesh scale factors')
    parser.add_argument(
        '--mesh-order',
        '-p',
        type=int,
        default=1,
        help='Mesh order')
    parser.add_argument(
        '--linear-heat',
        '-l',
        type=strtobool,
        default=True,
        help='If True, linear heat problem [True]')
    parser.add_argument(
        '--shape-scales',
        '-k',
        type=float,
        nargs='+',
        default=[1.],
        help='Shape scale factors')
    parser.add_argument(
        '--conductivities',
        '-c',
        type=float,
        nargs='+',
        default=CONDUCTIVITIES,
        help='Thermal conductivities')
    parser.add_argument(
        '--conductivity-mode',
        '-m',
        type=str,
        default='scalar',
        help='Thermal conductivity mode [\'scalar\', \'tensor\']')
    parser.add_argument(
        '--constant-tensor-conductivity',
        '-g',
        type=strtobool,
        default=False,
        help='If True, use constant tensor thermal conductivity [False]')
    parser.add_argument(
        '--nl-conductivity',
        '-b',
        type=float,
        nargs='+',
        default=None,
        help='Nonlinear thermal conductivities')
    parser.add_argument(
        '--n-repetition',
        '-n',
        type=int,
        default=3,
        help='The number of repetition.')
    parser.add_argument(
        '--timeout',
        '-t',
        type=int,
        default=600,
        help='Timeout of meshing in sec')
    args = parser.parse_args()

    generator = DataGenerator(**vars(args), seed=3)
    generator.generate()

    return


class DataGenerator():

    def __init__(
            self, input_cad_directories, *, output_directory=RAW_DIRECTORY,
            clscales=CLSCALES, linear_heat=True,
            conductivities=CONDUCTIVITIES, n_repetition=3, steepnesses=[1.],
            nl_conductivity=NL_CONDUCTIVITY,
            timeout=600, write_ucd=False, conductivity_scale=0.02,
            t_init_function=None, seed=None,
            shape_scales=[1.], mesh_order=1, meshdoctor=False,
            conductivity_mode='scalar', constant_tensor_conductivity=False):
        """Initialize DataGenerator object.

        Parameters
        ----------
        input_cad_directories: List[pathlib.Path]
            Input directory paths containing CAD STEP files.
        output_directory: pathlib.Path, optional
            The root of the output directory path. The default is
            Path('data/raw').
        clscales: List[float], optional
            List of clscales which determine scale of mesh. For more detail,
            see docs of gmsh. The default is (.1, .2, .3).
        linear_heat: bool, optional
            If True, thermal conductivity does not depend on
            temperature. The default is True.
        conductivities: List[float], optional
            List of thermal conductivities to be used linear_heat analysis.
            If linear_heat is False, this is ignored and nonlinear thermal
            conductivities are set at random. The default is
            np.arange(.5e-2, 1.5e-2 + 1e-5, .1e-2).
        n_repetition: int, optional
            The number of repetition to generate temperature field. The
            default is 3.
        steepnesses: List[float], optional
            List of steepnesses of temperature field. The default is
            (1.,).
        timeout: int, optional
            Maximum duration of meshing in second. If the limit exceeded, that
            meshing is regarded as failed. The default is 600.
        write_ucd: bool, optional
            If True, write AVS UCD file for the generated condition. The
            default is False.
        conductivity_scale: float, optional
            The scale of nonlinear thermal conductivity. The default is 0.02.
        t_init_function: collable, optional
            If provided, generate initial temperature with that function.
            It should accept only one argument, node position.
        seed: int, optional, optional
            If fed, set random seed to be used to create t_init.
        shape_scales: List[float], optional
            Scale of the generated shape. The default is [1.].
        mesh_order: int, optional
            The order of mesh. The default is 1.
        meshdoctor: bool, optional
            If True, perform MESHDOCTOR, to extract the largest connected
            component. The default is False.
        conductivity_mode: str, optional
            'scalar' or 'tensor'. The default is 'scalar'.
        constant_tensor_conductivity: bool, optional
            If True use constant conductivity for tensor conductivity
        """
        self.output_directory = output_directory
        self.clscales = clscales
        self.linear_heat = linear_heat
        self.conductivities = conductivities
        self.n_repetition = n_repetition
        self.steepnesses = steepnesses
        self.timeout = timeout
        self.write_ucd = write_ucd
        self.conductivity_scale = conductivity_scale
        self.seed = seed
        self.shape_scales = shape_scales
        self.mesh_order = mesh_order
        self.meshdoctor = meshdoctor
        self.nl_conductivity = nl_conductivity
        self.conductivity_mode = conductivity_mode
        self.constant_tensor_conductivity = constant_tensor_conductivity
        self.max_process = siml.util.determine_max_process()

        self.input_cad_files = [
            pathlib.Path(f) for f in np.unique(np.concatenate([
                glob(str(input_cad_directory / '**/*.step'), recursive=True)
                for input_cad_directory in input_cad_directories]))]

        if t_init_function is None:
            self.t_init_function = self._t_init_function
        else:
            self.t_init_function = t_init_function

        if self.constant_tensor_conductivity:
            c0 = self._generate_conductivity_array()
            self.raw_conductivity = np.concatenate([c0, c0 * .5])
            print(f"Use constant tensor conductivity: {c0}")
        return

    def generate(self):
        """Generate data, namely perform meshing and then add analysis
        conditions.
        """
        with multi.Pool(self.max_process) as pool:
            pool.map(self._generate_from_one_cad, self.input_cad_files)

    def _generate_from_one_cad(self, input_cad_file):
        for clscale in self.clscales:
            output_directory = siml.prepost.determine_output_directory(
                input_cad_file.parent, self.output_directory, 'external') \
                / f"clscale{clscale}"

            output_mesh_file = self.generate_mesh_if_needed(
                output_directory, input_cad_file, clscale=clscale)
            if output_mesh_file is None:
                continue

            for steepness in self.steepnesses:
                for shape_scale in self.shape_scales:
                    if self.linear_heat:
                        for conductivity in self.conductivities:
                            self.generate_analysis(
                                output_mesh_file, output_directory,
                                conductivity=conductivity,
                                steepness=steepness,
                                scale_factor=shape_scale)
                    else:
                        self.generate_analysis(
                            output_mesh_file, output_directory,
                            steepness=steepness,
                            scale_factor=shape_scale)
        return

    def generate_mesh_if_needed(
            self, output_directory, input_file, *, clscale=None, n_thread=1):
        """Generate mesh if meshing is not yet done.

        Parameters
        ----------
        output_directory: pathlib.Path
            Output directory path.
        input_file: pathlib.Path
            Input CAD file.
        clscale: float
            Parameter which determine scale of mesh. For more detail,
            see docs of gmsh. The default is None (meaning the default of
            gmsh.)
        n_thread: int, optional
            The number of thread to be used for meshing. The default is 1.

        Returns
        -------
        output_mesh_file: pathlib.Path
            If meshing is done successfully, the meshed VTK file path is
            returned. In case of meshdoctor=True, the format will be of
            FrontISTR msh.
            Otherwise, None is returned, meaning meshing is already
            done or meshing failed.
        """
        output_mesh_file = output_directory / 'mesh.vtk'
        if (output_directory / MESHING_SUCCESS_FILE).exists():
            print(
              f"Already meshed for: {output_directory}\n"
              'Skip meshing.')
            return output_mesh_file

        if clscale is None:
            str_clscale = ''
        else:
            str_clscale = f"-clscale {clscale}"

        output_directory.mkdir(parents=True, exist_ok=True)
        output_mesh_file = output_directory / 'mesh.vtk'
        mesh_log_file = output_directory / 'gmsh.log'
        try:
            sp = subprocess.run(
                f"timeout {self.timeout} mpirun -np 1 --allow-run-as-root "
                f"gmsh {input_file} "
                f"-o {output_mesh_file} -format vtk -order {self.mesh_order} "
                f"-0 -3 {str_clscale} -nt {n_thread} "
                f"2>&1 | tee {mesh_log_file}",
                shell=True, check=True)
            print(sp)
        except subprocess.CalledProcessError:
            print(
              f"Meshing failed for: {output_directory}\n"
              'Go to next setting.')
            if output_mesh_file.exists():
                output_mesh_file.unlink()
                # output_mesh_file.unlink(missing_ok=True)  # For python3.8
            return None

        if self.meshdoctor:
            output_mesh_file = self._meshdoctor(output_mesh_file)
            if output_mesh_file is None:
                return None

        (output_directory / MESHING_SUCCESS_FILE).touch()

        return output_mesh_file

    def _meshdoctor(self, input_vtk_file):
        if not input_vtk_file.is_file():
            return

        fem_data = femio.FEMData.read_files('vtk', input_vtk_file)
        fem_data.settings['solution_type'] = 'MESHDOCTOR'
        fem_data.element_groups = {'ALL': fem_data.elements.ids}
        fem_data.sections.update_data(
            'MAT_ALL', {'TYPE': 'SOLID', 'EGRP': 'ALL'})
        fem_data.materials.reset()
        fem_data.materials.update_data(
            'MAT_ALL', {
                'Young_modulus': np.array([[1.]]),
                'Poisson_ratio': np.array([[0.]]),
                'density': np.array([[1.]]),
            })

        output_directory = input_vtk_file.parent
        fem_data.write('fistr', output_directory / 'meshdoctor')
        output_mesh_file = output_directory / 'output/mesh.msh'
        try:
            sp = subprocess.run(
                f"cd {output_directory} && fistr1", shell=True, check=True)
            print(sp)
        except subprocess.CalledProcessError:
            print(
              f"MESHDOCTOR failed for: {output_mesh_file}\n"
              'Go to next setting.')
            if output_mesh_file.exists():
                output_mesh_file.unlink()
            return None

        return output_mesh_file

    def generate_analysis(
            self, input_mesh, output_base_directory, *,
            conductivity=1., steepness=1., scale_factor=1.):
        """Generate FrontISTR HEAT simulation input data.

        Parameters
        ----------
        input_mesh: pathlib.Path
            Input mesh VTK file or FrontISTR msh file (when meshdoctor=True).
        output_base_directory: pathlib.Path
            Output directory.
        n_repetition: int, optional [5]
            The number of repetitions to randomly set initial condition and
            material property.
        conductivity: float, optional [1.]
            Heat conductivity.
        steepness: float, optional [1.]
            The steepness of temperature field.
        scale_factor: float, optional [1.]
            Control parameter for the scale of the shape.

        Returns
        -------
        None
        """
        if not input_mesh.is_file():
            return

        if self.meshdoctor:
            fem_data = femio.FEMData.read_files(
                'fistr', [input_mesh], read_mesh_only=True)
            fem_data.elemental_data.reset()
            fem_data.materials.reset()
            fem_data.sections.reset()
        else:
            try:
                fem_data = femio.FEMData.read_files('vtk', [input_mesh])
            except KeyError:
                print(f"No solid mesh found for: {input_mesh}")
                return

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Scale mesh
        fem_data = self._scale_mesh(fem_data, scale_factor=scale_factor)

        # Basic setting
        fem_data = self._add_basic_setting(fem_data)

        for i_repetition in range(self.n_repetition):
            output_directory = self._determine_output_directory(
                output_base_directory, i_repetition, conductivity, steepness)
            generated_fem_data = self._add_condition(
                fem_data, output_directory, conductivity=conductivity,
                steepness=steepness)
            if generated_fem_data is not None:
                self._write(generated_fem_data, output_directory)
                (output_directory / GENERATION_SUCCESS_FILE).touch()
        return

    def _t_init_function(self, nodes):
        t_init = np.mean(
            [
                (2 * np.random.rand() - 1.) * np.cos(
                    i_x * nodes[:, 0]
                    + i_y * nodes[:, 1]
                    + (n_degree - i_x - i_y) * nodes[:, 2]
                    + np.random.rand() * 2 * np.pi)
                for n_degree in range(
                    MIN_FOURIER_DEGREE, MAX_FOURIER_DEGREE + 1)
                for i_x in range(n_degree + 1)
                for i_y in range(n_degree - i_x + 1)],
            axis=0)
        return t_init

    def _determine_output_directory(
            self, output_base_directory, i_repetition, conductivity,
            steepness):
        if self.linear_heat:
            output_directory = output_base_directory \
                / (
                    f"steepness{steepness:.1f}_conductivity{conductivity:.3e}"
                    + f"_rep{i_repetition}")
        else:
            output_directory = output_base_directory \
                / (f"steepness{steepness:.1f}_rep{i_repetition}")
        return output_directory

    def _write(self, fem_data, output_directory):
        base_name = 'heat'
        fem_data.write('fistr', output_directory / base_name)
        if self.write_ucd:
            fem_data.write('ucd', output_directory / f"{base_name}.inp")
        return

    def _add_condition(
            self, fem_data, output_directory, *,
            conductivity=1., steepness=1.):
        if (output_directory / GENERATION_SUCCESS_FILE).exists():
            print(
              f"Already generated for: {output_directory}\n"
              'Go to next setting.')
            return None

        if self.linear_heat:
            conductivity_array = np.array([[conductivity, 0.]])
        else:
            if self.conductivity_mode == 'tensor':
                if self.constant_tensor_conductivity:
                    raw_conductivity = self.raw_conductivity
                else:
                    # Assume tensor direction does not change, but the
                    # norm decreases with temperature
                    c0 = self._generate_conductivity_array()
                    raw_conductivity = np.concatenate([c0, c0 * .5])
                conductivity_array = np.array([[
                    np.concatenate([
                        raw_conductivity,
                        np.array([[-1.], [1.]])], axis=1), 0
                ]], dtype=object)[:, 0]
            elif self.conductivity_mode == 'scalar':
                if self.nl_conductivity is None:
                    conductivity_array = np.array([[
                        np.concatenate([
                            (.5 * np.random.rand(2, 1) + .5)
                            * self.conductivity_scale,  # [1/2 * scale, scale]
                            np.array([[-1.], [1.]])], axis=1), 0
                    ]], dtype=object)[:, 0]
                else:
                    conductivity_array = np.array([[
                        np.concatenate([
                            np.array(self.nl_conductivity)[:, None],
                            np.array([[-1.], [1.]])], axis=1), 0
                        ]], dtype=object)[:, 0]
            else:
                raise ValueError(
                    f"Unexpected conductivity mode: {self.conductivity_mode}")

        t_init = self.t_init_function(fem_data.nodes.data)

        fem_data.nodal_data.set_attribute_data(
            'INITIAL_TEMPERATURE',
            self._scale_temperature(t_init, steepness=steepness),
            allow_overwrite=True)

        # Material
        if self.conductivity_mode == 'scalar':
            fem_data.materials.update_data(
                'MAT_ALL', {
                    'density': np.array([[1., 0.]]),
                    'specific_heat': np.array([[1., 0.]]),
                    'thermal_conductivity': conductivity_array},
                allow_overwrite=True)
        elif self.conductivity_mode == 'tensor':
            fem_data.materials.update_data(
                'MAT_ALL', {
                    'density': np.array([[1., 0.]]),
                    'specific_heat': np.array([[1., 0.]]),
                    'thermal_conductivity': np.array([[1., 0.]]),
                    'thermal_conductivity_full': conductivity_array},
                allow_overwrite=True)

        return fem_data

    def _generate_conductivity_array(self):
        eigenvalues = np.random.rand(3) * self.conductivity_scale
        v1 = self._normalize(np.random.rand(3))
        tmp_v2 = np.random.rand(3)
        v3 = self._normalize(np.cross(v1, tmp_v2))
        v2 = np.cross(v3, v1)
        rotation_matrix = np.stack([v1, v2, v3])
        conductivity_mat = rotation_matrix @ np.diag(
            eigenvalues) @ rotation_matrix.T
        conductivity_array = femio.FEMData.convert_symmetric_matrix2array(
            None, conductivity_mat, to_engineering=False)
        return conductivity_array

    def _normalize(self, x):
        return x / np.linalg.norm(x)

    def _scale_mesh(self, fem_data, scale_factor):
        raw_nodes = fem_data.nodes.data
        lengths = np.max(raw_nodes, axis=0) - np.min(raw_nodes, axis=0)
        fem_data.nodal_data.overwrite(
            'NODE', raw_nodes / np.max(lengths) * scale_factor)
        scaled_nodes = raw_nodes / np.max(lengths) * scale_factor
        fem_data.nodes.update_data(scaled_nodes)
        fem_data.nodal_data.set_attribute_data(
            'NODE', scaled_nodes, allow_overwrite=True)
        return fem_data

    def _add_basic_setting(self, fem_data):
        fem_data.settings['solution_type'] = 'HEAT'
        fem_data.settings['heat'] = np.array([[0.01, 1.]])

        elemental_ids = fem_data.elements.ids
        fem_data.settings['frequency'] = 10
        fem_data.settings['beta'] = 1.
        # fem_data.settings['write_visual'] = False
        fem_data.element_groups = {'ALL': elemental_ids}
        fem_data.sections.update_data(
            'MAT_ALL', {'TYPE': 'SOLID', 'EGRP': 'ALL'})
        return fem_data

    def _scale_temperature(self, data, steepness):
        min_data = np.min(data)
        max_data = np.max(data)

        # Scales from -1 to 1
        data = (2 * data - (max_data + min_data)) / (max_data - min_data)

        # Filter data
        scale_data = np.tanh(steepness * data) / np.tanh(steepness * 1)

        return scale_data


if __name__ == '__main__':
    main()
