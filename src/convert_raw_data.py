import argparse
from datetime import datetime as dt
from distutils.util import strtobool
import pathlib
import re
import subprocess

import numpy as np
import siml
from scipy import sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, overwrite existing data [False]')
    parser.add_argument(
        '-n', '--read-npy',
        type=strtobool,
        default=1,
        help='If True, read .npy files instead of original files '
        'if exists [True]')
    parser.add_argument(
        '-r', '--recursive',
        type=strtobool,
        default=1,
        help='If True, process directory recursively [True]')
    parser.add_argument(
        '-e', '--elemental',
        type=strtobool,
        default=0,
        help='If True, create also elemental features [False]')
    parser.add_argument(
        '-w', '--write-ucd',
        type=strtobool,
        default=0,
        help='If True, write AVS UCD file of converted interim data [False]')
    parser.add_argument(
        '-p', '--load-parent',
        type=strtobool,
        default=0,
        help='If True, load parent data for heat analysis [False]')
    parser.add_argument(
        '-l', '--light',
        type=strtobool,
        default=0,
        help='If True, compute heavy features [False]')
    args = parser.parse_args()

    conversion_function = HeatConversionFuncionCreator(
        create_elemental=args.elemental, light=args.light)
    filter_function = filter_function_heat
    raw_converter = siml.prepost.RawConverter.read_settings(
        args.settings_yaml,
        conversion_function=conversion_function,
        filter_function=filter_function,
        force_renew=args.force_renew,
        recursive=args.recursive,
        to_first_order=True,
        write_ucd=args.write_ucd,
        read_npy=args.read_npy, read_res=True)
    raw_converter.convert()
    print('success')


class ConversionFuncionCreator():

    def process_geometry(self, fem_data, light=False):
        elemental_volume = fem_data.calculate_element_volumes(
            raise_negative_volume=False, return_abs_volume=False)
        if np.any(elemental_volume < 0.):
            # Pass to filter function
            return {'elemental_volume': elemental_volume}

        filter_ = fem_data.filter_first_order_nodes()
        node = fem_data.nodal_data.get_attribute_data('node')[filter_]

        nodal_mean_volume = fem_data.convert_elemental2nodal(
            elemental_volume, mode='mean', raise_negative_volume=False)
        nodal_effective_volume = fem_data.convert_elemental2nodal(
            elemental_volume, mode='effective', raise_negative_volume=False)

        nodal_grad_x_2, nodal_grad_y_2, nodal_grad_z_2 = \
            fem_data.calculate_spatial_gradient_adjacency_matrices(
                'nodal', n_hop=2, moment_matrix=True)

        dict_data = {
            'node': node,
            'nodal_mean_volume': nodal_mean_volume,
            'elemental_volume': elemental_volume,
            'nodal_effective_volume': nodal_effective_volume,
            'nodal_grad_x_2': nodal_grad_x_2,
            'nodal_grad_y_2': nodal_grad_y_2,
            'nodal_grad_z_2': nodal_grad_z_2,
        }

        if not light:
            nodal_adj_2 = fem_data.calculate_n_hop_adj(
                mode='nodal', n_hop=2).tocoo()
            nodal_nadj_2 = normalize_adjacency_matrix(nodal_adj_2)

            nodal_adj_10 = fem_data.calculate_n_hop_adj(
                mode='nodal', n_hop=10).tocoo()
            nodal_nadj_10 = normalize_adjacency_matrix(
                nodal_adj_10)

            nodal_adj = fem_data.calculate_adjacency_matrix_node()
            nodal_nadj = normalize_adjacency_matrix(nodal_adj)

            nodal_grad_x_1, nodal_grad_y_1, nodal_grad_z_1 = \
                fem_data.calculate_spatial_gradient_adjacency_matrices(
                    'nodal', n_hop=1, moment_matrix=True)

            nodal_adj_4 = fem_data.calculate_n_hop_adj(
                mode='nodal', n_hop=4).tocoo()
            nodal_nadj_4 = normalize_adjacency_matrix(nodal_adj_4)

            nodal_adj_5 = fem_data.calculate_n_hop_adj(
                mode='nodal', n_hop=5).tocoo()
            nodal_nadj_5 = normalize_adjacency_matrix(nodal_adj_5)
            nodal_grad_x_5, nodal_grad_y_5, nodal_grad_z_5 = \
                fem_data.calculate_spatial_gradient_adjacency_matrices(
                    'nodal', n_hop=5, moment_matrix=True)

            dict_data.update({
                'nodal_adj': nodal_adj, 'nodal_nadj': nodal_nadj,
                'nodal_adj_2': nodal_adj_2, 'nodal_nadj_2': nodal_nadj_2,
                'nodal_adj_4': nodal_adj_4, 'nodal_nadj_4': nodal_nadj_4,
                'nodal_adj_5': nodal_adj_5, 'nodal_nadj_5': nodal_nadj_5,
                'nodal_adj_10': nodal_adj_10, 'nodal_nadj_10': nodal_nadj_10,
                'nodal_grad_x_1': nodal_grad_x_1,
                'nodal_grad_y_1': nodal_grad_y_1,
                'nodal_grad_z_1': nodal_grad_z_1,
                'nodal_grad_x_5': nodal_grad_x_5,
                'nodal_grad_y_5': nodal_grad_y_5,
                'nodal_grad_z_5': nodal_grad_z_5,
            })

        return dict_data


class HeatConversionFuncionCreator(ConversionFuncionCreator):

    def __init__(
            self, create_elemental=False, convert_answer=True, light=False):
        self.create_elemental = create_elemental
        self.convert_answer = convert_answer
        self.light = light
        return

    def __call__(self, fem_data, raw_directory):
        print('============ Start preprocessing ============')
        start_time = dt.now()

        dict_data = self.process_geometry(fem_data, light=self.light)
        filter_ = fem_data.filter_first_order_nodes()

        if 'thermal_conductivity_full' in fem_data.elemental_data:
            raw_conductivity = fem_data.elemental_data.get_attribute_data(
                'thermal_conductivity_full')
            # NOTE: Extract the zero th component assuming the 1st component
            #       is linear dependent
            elemental_thermal_conductivity_array = np.stack([
                c[0, :-1] for c in raw_conductivity[:, 0]])
            elemental_thermal_conductivity \
                = fem_data.convert_array2symmetric_matrix(
                    elemental_thermal_conductivity_array,
                    from_engineering=False)[:, :, :, None]
            nodal_thermal_conductivity_array \
                = fem_data.convert_elemental2nodal(
                    elemental_thermal_conductivity_array, mode='mean',
                    raise_negative_volume=False)
            nodal_thermal_conductivity \
                = fem_data.convert_array2symmetric_matrix(
                    nodal_thermal_conductivity_array, from_engineering=False)[
                        :, :, :, None]
        else:
            raw_conductivity = fem_data.elemental_data.get_attribute_data(
                'thermal_conductivity')
            if raw_conductivity.dtype == 'object':
                elemental_thermal_conductivity = np.stack([
                    np.ravel(c[:, 0]) for c
                    in raw_conductivity[:, 0]])
            else:
                elemental_thermal_conductivity = raw_conductivity

            nodal_thermal_conductivity = fem_data.convert_elemental2nodal(
                elemental_thermal_conductivity, mode='mean',
                raise_negative_volume=False)

        finish_time = dt.now()
        print('============ Finish preprocessing ============')
        print(f"Preprocess time: {finish_time - start_time}")

        # NOTE: Since the following data is not necessary for inference,
        #       we exclude from the preprocessing time
        if not self.light:
            nodal_laplacian_1 = self.create_spatial_laplacian(
                dict_data['nodal_grad_x_1'],
                dict_data['nodal_grad_y_1'],
                dict_data['nodal_grad_z_1'])

            nodal_laplacian_2 = self.create_spatial_laplacian(
                dict_data['nodal_grad_x_2'],
                dict_data['nodal_grad_y_2'],
                dict_data['nodal_grad_z_2'])

            nodal_laplacian_5 = self.create_spatial_laplacian(
                dict_data['nodal_grad_x_5'],
                dict_data['nodal_grad_y_5'],
                dict_data['nodal_grad_z_5'])
            dict_data.update({
                'nodal_laplacian_1': nodal_laplacian_1,
                'nodal_laplacian_2': nodal_laplacian_2,
                'nodal_laplacian_5': nodal_laplacian_5,
            })

        nodal_t_0 = fem_data.nodal_data.get_attribute_data(
            'INITIAL_TEMPERATURE')[filter_]
        global_thermal_conductivity = np.mean(
            elemental_thermal_conductivity, keepdims=True, axis=0)

        dict_data.update({
            'nodal_thermal_conductivity': nodal_thermal_conductivity,
            'nodal_t_0': nodal_t_0,
            'global_thermal_conductivity': global_thermal_conductivity,
        })
        if self.convert_answer:
            temperatures = fem_data.nodal_data.get_attribute_data(
                'TEMPERATURE')
            dict_t_data = {
                f"nodal_t_{step}": t[filter_] for step, t in zip(
                    fem_data.settings['time_steps'], temperatures)}
            max_timestep = max(fem_data.settings['time_steps'])
            dict_data.update(dict_t_data)
            dict_data.update({
                'nodal_t_diff':
                dict_data[f"nodal_t_{max_timestep}"] - dict_data['nodal_t_0']})

        if self.create_elemental:
            elemental_dict_data = self.convert_elemental(
                fem_data, dict_data)
            dict_data.update(elemental_dict_data)

        dict_data.update(self.create_tv(dict_data))

        return dict_data

    def create_tv(self, dict_data):
        """Calculate T * dV"""
        additional_dict = {}
        for key, value in dict_data.items():
            match = re.search(r'nodal_t_(\d+)', key)
            if match is None:
                continue
            number_step = int(match.groups()[0])
            additional_dict.update({
                f"nodal_tv_{number_step}":
                value * dict_data['nodal_effective_volume']})
        return additional_dict

    def create_spatial_laplacian(self, grad_x, grad_y, grad_z):
        return (
            grad_x.dot(grad_x) + grad_y.dot(grad_y) + grad_z.dot(grad_z)
        ).tocoo() / 6

    def create_elemental_dict_data(self, fem_data, dict_data):
        elemental_adj, _ = fem_data.calculate_adjacency_matrix_element()
        elemental_nadj = normalize_adjacency_matrix(elemental_adj)

        elemental_grad_x_2, elemental_grad_y_2, elemental_grad_z_2 = \
            fem_data.calculate_spatial_gradient_adjacency_matrices(
                'elemental', n_hop=2)
        elemental_grad_x_5, elemental_grad_y_5, elemental_grad_z_5 = \
            fem_data.calculate_spatial_gradient_adjacency_matrices(
                'elemental', n_hop=5)

        dict_elemental_t_data = {}
        for nodal_data_name, nodal_data_value in dict_data.items():
            if nodal_data_name in 'nodal_t_':
                elemental_t_step, elemental_ave_t_step = \
                    self.convert_nodal2elemental(fem_data, nodal_data_value)
                dict_elemental_t_data.update({
                    nodal_data_name.replace('nodal', 'elemental'):
                    elemental_t_step,
                    nodal_data_name.replace('nodal', 'elemental_ave'):
                    elemental_ave_t_step})

        elemental_dict_data = {
            'elemental_adj': elemental_adj,
            'elemental_nadj': elemental_nadj,
            'elemental_grad_x_2': elemental_grad_x_2,
            'elemental_grad_y_2': elemental_grad_y_2,
            'elemental_grad_z_2': elemental_grad_z_2,
            'elemental_grad_x_5': elemental_grad_x_5,
            'elemental_grad_y_5': elemental_grad_y_5,
            'elemental_grad_z_5': elemental_grad_z_5,
        }
        elemental_dict_data.update(dict_elemental_t_data)
        return elemental_dict_data

    def convert_nodal2elemental(self, fem_data, data):
        elemental_data = fem_data.convert_nodal2elemental(
            data, ravel=True)[:, :4]
        elemental_ave_data = np.mean(elemental_data, axis=1, keepdims=True)
        return elemental_data, elemental_ave_data


class ThermalFilterFunctionCreator():

    def __init__(self, max_abs_strain):
        self.max_abs_strain = max_abs_strain
        if self.max_abs_strain > .1:
            self.max_relative_rmse_thermal = 3.
        else:
            self.max_relative_rmse_thermal = .8
        return

    def __call__(self, fem_data, raw_directory, dict_data):
        # Filter by heat data
        filter_by_heat = filter_function_heat(
            fem_data, raw_directory, dict_data)
        if filter_by_heat is False:
            return False

        # Filter by residual value of the thermal computation
        str_residual = subprocess.run(
            f"grep residual {raw_directory / 'FSTR.msg'} | rev "
            '| cut -d: -f1 | rev',
            shell=True, capture_output=True).stdout
        try:
            float_residual = float(str_residual)
        except ValueError:
            print(f"Residual parse error: {str_residual} in {raw_directory}")
            return False
        if float_residual > 1e-8:
            return False

        # Filter by thermal computation strain
        max_abs_strain = np.max(np.abs(dict_data['nodal_strain_array']))
        if max_abs_strain > self.max_abs_strain:
            print(
                f"Max abs strain too learge: {max_abs_strain} "
                f"for {raw_directory}")
            return False
        if 'nodal_ltec' in dict_data:
            ltec = dict_data['nodal_ltec']
        else:
            ltec = dict_data['nodal_ltec_array']
        relative_rmse_thermal = np.mean((
            ltec * dict_data['nodal_t_diff']
            - dict_data['nodal_strain_array'])**2)**.5 \
            / np.mean(dict_data['nodal_strain_array']**2)**.5
        if not np.isfinite(relative_rmse_thermal) \
                or relative_rmse_thermal > self.max_relative_rmse_thermal:
            print(
                f"Relative RMSE of thermal invalid: {relative_rmse_thermal} "
                f"for {raw_directory}")
            return False

        return True


def filter_function_heat(fem_data, raw_directory, dict_data):
    if np.any(dict_data['elemental_volume'] < 0.):
        print(f"Negative volume found. Skipped: {raw_directory}")
        return False

    if 'nodal_t_100' not in dict_data:
        return True

    max_t_100 = np.max(np.abs(dict_data['nodal_t_100']))
    max_t_0 = np.max(np.abs(dict_data['nodal_t_0']))
    print(f"max_t_0: {max_t_0:.3e}")
    print(f" max_t_100: {max_t_100:.3e}")
    if max_t_100 > max_t_0:
        print(f"Heat analysis not converged. Skipped: {raw_directory}")
        return False

    return True


def normalize_adjacency_matrix(adj):
    print(f"to_coo adj: {dt.now()}")
    adj = sp.coo_matrix(adj).astype(float)
    diag = sp.diags(adj.diagonal())
    adj_w_selfloop = adj - diag + sp.eye(adj.shape[0])
    print(f"sum raw: {dt.now()}")
    rowsum = np.array(adj_w_selfloop.sum(1))
    print(f"invert d: {dt.now()}")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(f"making diag: {dt.now()}")
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print(f"calculating norm: {dt.now()}")
    return adj_w_selfloop.dot(
        d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


if __name__ == '__main__':
    main()
