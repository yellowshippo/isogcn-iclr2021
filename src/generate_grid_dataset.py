"""Generate raw data."""

import argparse
import multiprocessing as multi
import pathlib
import random

import femio
import numpy as np
import siml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_directory',
        type=pathlib.Path,
        help='Output base direcoty')
    parser.add_argument(
        '--n-repetition',
        '-n',
        type=int,
        default=3,
        help='The number of repetition [3]')
    parser.add_argument(
        '--min-n_element',
        '-j',
        type=int,
        default=10,
        help='The minimum number of elements [10]')
    parser.add_argument(
        '--max-n_element',
        '-k',
        type=int,
        default=100,
        help='The maximum number of elements [10]')
    parser.add_argument(
        '--min-fourier-degree',
        '-f',
        type=int,
        default=1,
        help='The minimum Fourier degree [1]')
    parser.add_argument(
        '--max-fourier-degree',
        '-g',
        type=int,
        default=10,
        help='The maximum Fourier degree [10]')
    parser.add_argument(
        '--max-process',
        '-p',
        type=int,
        default=None,
        help='If fed, set the maximum # of processes')
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        default=None,
        help='If fed, set random seed')
    args = parser.parse_args()

    generator = GridDataGenerator(**vars(args))
    generator.generate()

    return


class GridDataGenerator:

    DIM = 2

    def __init__(
            self, output_directory, *,
            edge_length=1., n_repetition=3, seed=None,
            min_n_element=10, max_n_element=100,
            max_fourier_degree=10, min_fourier_degree=1, max_process=None):
        self.output_directory = pathlib.Path(output_directory)
        self.edge_length = edge_length
        self.n_repetition = n_repetition
        self.seed = seed
        self.min_n_element = min_n_element
        self.max_n_element = max_n_element
        self.max_fourier_degree = max_fourier_degree
        self.min_fourier_degree = min_fourier_degree
        self.max_process = siml.util.determine_max_process(max_process)
        self.max_length = self.max_n_element * self.edge_length

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return

    def generate(self, n_elements=None):
        """Create grid graph data.

        Parameters
        ----------
        n_elements: List[int]
            The number of elements for x, y, and z direction.
        """
        self.n_elements = n_elements
        if self.n_elements is not None:
            if len(self.n_elements) != self.DIM:
                raise ValueError(
                    f"len(self.n_elements) should be {self.DIM} "
                    f"(given: {len(self.n_elements)}")

        with multi.Pool(self.max_process) as pool:
            pool.map(self._generate_one_data, list(range(self.n_repetition)))
        return

    def _generate_one_data(self, i_data):
        if self.n_elements is None:
            n_x_element = random.randint(
                self.min_n_element, self.max_n_element)
            n_y_element = random.randint(
                self.min_n_element, self.max_n_element)
        else:
            n_x_element, n_y_element = self.n_elements
        n_z_element = 1

        fem_data = self.generate_grid(
            n_x_element, n_y_element, n_z_element)
        target_dict_data = self.add_gradient_data(fem_data)
        dict_data = self.extract_feature(fem_data, target_dict_data)

        output_directory = self.output_directory / str(i_data)
        self.save(output_directory, dict_data, fem_data)
        return

    def generate_grid(self, n_x_element=10, n_y_element=10, n_z_element=1):

        # Generate nodes
        # NOTE: Swap x and y to order positions nicely
        y, x, z = np.meshgrid(
            np.linspace(
                0., n_y_element * self.edge_length, n_y_element + 1),
            np.linspace(
                0., n_x_element * self.edge_length, n_x_element + 1),
            np.linspace(
                0., n_z_element * self.edge_length, n_z_element + 1),
        )
        raw_nodes = np.stack([np.ravel(x), np.ravel(y), np.ravel(z)], axis=-1)
        node_ids = np.arange(len(raw_nodes)) + 1  # Make 1 origin
        nodes = femio.FEMAttribute(
            'NODE', ids=node_ids, data=raw_nodes)

        # Generate elements
        n_x = n_x_element + 1
        n_y = n_y_element + 1
        n_z = n_z_element + 1
        rim_x = self.edge_length * n_x_element
        rim_y = self.edge_length * n_y_element
        rim_z = self.edge_length * n_z_element
        raw_elements = np.array([
            [
                i, i + n_y * n_z, i + n_y * n_z + n_z, i + n_z,
                i + 1, i + 1 + n_y * n_z, i + 1 + n_y * n_z + n_z, i + 1 + n_z]
            for i in range(n_x * n_y * n_z)
            if raw_nodes[i, 0] < rim_x - 1.e-5
            and raw_nodes[i, 1] < rim_y - 1.e-5
            and raw_nodes[i, 2] < rim_z - 1.e-5]) + 1  # Make 1 origin

        element_ids = np.arange(len(raw_elements)) + 1  # Make 1 origin
        elements = femio.FEMElementalAttribute(
            'ELEMENT', {
                'hex':
                femio.FEMAttribute('hex', ids=element_ids, data=raw_elements)})

        fem_data = femio.FEMData(nodes=nodes, elements=elements)
        return fem_data

    def add_gradient_data(self, fem_data):
        nodes = fem_data.nodes.data
        scalar_field, gradient, hessian \
            = self._calculate_fourier_component_and_gradients(nodes)
        laplacian = np.trace(hessian, axis1=1, axis2=2)[:, None]
        dict_data = {
            'scalar_field': scalar_field, 'gradient': gradient,
            'hessian': hessian, 'laplacian': laplacian}

        hessian_array = fem_data.convert_symmetric_matrix2array(
            hessian, to_engineering=False)
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {
                'scalar_field': scalar_field, 'gradient': gradient,
                'hessian_array': hessian_array, 'laplacian': laplacian})

        return dict_data

    def _calculate_fourier_component_and_gradients(self, nodes):
        n_degree = random.randint(
            self.min_fourier_degree, self.max_fourier_degree)
        coeffs = [
            random.random()
            for i_degree in range(self.min_fourier_degree, n_degree + 1)
            for i_x in range(i_degree + 1)]
        phase_shifts = [
            random.random() * 2 * np.pi
            for i_degree in range(self.min_fourier_degree, n_degree + 1)
            for i_x in range(i_degree + 1)]
        m = self.min_fourier_degree

        raw_scalar_field = np.mean(
            [
                coeffs[i_degree - m + i_x] * np.cos(
                    (i_x * nodes[:, 0] + (i_degree - i_x) * nodes[:, 1])
                    * 2 * np.pi / self.max_length
                    + phase_shifts[i_degree - m + i_x])
                for i_degree in range(self.min_fourier_degree, n_degree + 1)
                for i_x in range(i_degree + 1)],
            axis=0)
        scaling_factor = 1 / np.max(np.abs(raw_scalar_field))
        scalar_field = raw_scalar_field * scaling_factor
        gradient = np.mean(
            [
                - coeffs[i_degree - m + i_x] * np.sin(
                    (i_x * nodes[:, 0] + (i_degree - i_x) * nodes[:, 1])
                    * 2 * np.pi / self.max_length
                    + phase_shifts[i_degree - m + i_x])[:, None]
                * np.array([i_x, i_degree - i_x, 0])
                * 2 * np.pi / self.max_length
                for i_degree in range(self.min_fourier_degree, n_degree + 1)
                for i_x in range(i_degree + 1)],
            axis=0) * scaling_factor
        hessian = np.mean(
            [
                - coeffs[i_degree - m + i_x]
                * np.einsum(
                    'i,jk->ijk',
                    np.cos(
                        (i_x * nodes[:, 0] + (i_degree - i_x) * nodes[:, 1])
                        * 2 * np.pi / self.max_length
                        + phase_shifts[i_degree - m + i_x]),
                    np.array([
                        [i_x**2, i_x * (i_degree - i_x), 0],
                        [(i_degree - i_x) * i_x, (i_degree - i_x)**2, 0],
                        [0, 0, 0],
                    ]))
                * (2 * np.pi / self.max_length)**2
                for i_degree in range(self.min_fourier_degree, n_degree + 1)
                for i_x in range(i_degree + 1)],
            axis=0) * scaling_factor

        return scalar_field, gradient, hessian

    def extract_feature(self, fem_data, target_dict_data):
        scalar_field = target_dict_data['scalar_field'][:, None]
        gradient = target_dict_data['gradient'][..., None]
        hessian = target_dict_data['hessian'][..., None]
        laplacian = target_dict_data['laplacian']

        nodal_adj_10 = fem_data.calculate_n_hop_adj(
            mode='nodal', n_hop=10).tocoo()
        nodal_nadj_10 = siml.prepost.normalize_adjacency_matrix(nodal_adj_10)

        nodal_adj = fem_data.calculate_adjacency_matrix_node()
        nodal_nadj = siml.prepost.normalize_adjacency_matrix(nodal_adj)

        node = fem_data.nodal_data.get_attribute_data('node')

        nodal_grad_x_1, nodal_grad_y_1, nodal_grad_z_1 = \
            fem_data.calculate_spatial_gradient_adjacency_matrices(
                'nodal', n_hop=1, consider_volume=False)

        nodal_adj_2 = fem_data.calculate_n_hop_adj(
            mode='nodal', n_hop=2).tocoo()
        nodal_nadj_2 = siml.prepost.normalize_adjacency_matrix(nodal_adj_2)
        nodal_grad_x_2, nodal_grad_y_2, nodal_grad_z_2 = \
            fem_data.calculate_spatial_gradient_adjacency_matrices(
                'nodal', n_hop=2, consider_volume=False)

        nodal_adj_4 = fem_data.calculate_n_hop_adj(
            mode='nodal', n_hop=4).tocoo()
        nodal_nadj_4 = siml.prepost.normalize_adjacency_matrix(nodal_adj_4)

        nodal_adj_5 = fem_data.calculate_n_hop_adj(
            mode='nodal', n_hop=5).tocoo()
        nodal_nadj_5 = siml.prepost.normalize_adjacency_matrix(nodal_adj_5)
        nodal_grad_x_5, nodal_grad_y_5, nodal_grad_z_5 = \
            fem_data.calculate_spatial_gradient_adjacency_matrices(
                'nodal', n_hop=5, consider_volume=False)

        dict_data = {
            'scalar_field': scalar_field, 'gradient': gradient,
            'hessian': hessian, 'laplacian': laplacian,
            'node': node,
            'nodal_adj': nodal_adj, 'nodal_nadj': nodal_nadj,
            'nodal_adj_2': nodal_adj_2, 'nodal_nadj_2': nodal_nadj_2,
            'nodal_adj_4': nodal_adj_4, 'nodal_nadj_4': nodal_nadj_4,
            'nodal_adj_5': nodal_adj_5, 'nodal_nadj_5': nodal_nadj_5,
            'nodal_adj_10': nodal_adj_10, 'nodal_nadj_10': nodal_nadj_10,
            'nodal_grad_x_1': nodal_grad_x_1,
            'nodal_grad_y_1': nodal_grad_y_1,
            'nodal_grad_z_1': nodal_grad_z_1,
            'nodal_grad_x_2': nodal_grad_x_2,
            'nodal_grad_y_2': nodal_grad_y_2,
            'nodal_grad_z_2': nodal_grad_z_2,
            'nodal_grad_x_5': nodal_grad_x_5,
            'nodal_grad_y_5': nodal_grad_y_5,
            'nodal_grad_z_5': nodal_grad_z_5,
        }

        return dict_data

    def save(self, output_directory, dict_data, fem_data):
        siml.prepost.save_dict_data(output_directory, dict_data)
        fem_data.save(output_directory)
        fem_data.write('ucd', output_directory / 'mesh.inp')
        (output_directory / 'converted').touch()
        return


if __name__ == '__main__':
    main()
