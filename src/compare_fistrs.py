import argparse
from distutils.util import strtobool
import pathlib
import re
import subprocess

import femio
import numpy as np
import siml

import infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'fistr_directory',
        type=pathlib.Path,
        help='FrontISTR result directory.')
    parser.add_argument(
        'ref_fistr_directory',
        type=pathlib.Path,
        help='FrontISTR result directory for reference data.')
    args = parser.parse_args()

    fem_data = femio.FEMData.read_directory(
        'fistr', args.fistr_directory, read_npy=False)
    ref_fem_data = femio.FEMData.read_directory(
        'fistr', args.ref_fistr_directory)

    # Extract the latest temperature
    temperature = fem_data.nodal_data.get_attribute_data('TEMPERATURE')
    ref_temperature = ref_fem_data.nodal_data.get_attribute_data('TEMPERATURE')

    mse, std_error = infer.calculate_loss_stats(
        temperature, ref_temperature)

    nodal_effective_volume = fem_data.convert_elemental2nodal(
        fem_data.calculate_element_volumes(), mode='effective')
    l2_error, l2_std_error = infer.calculate_l2_stats(
        temperature, ref_temperature, nodal_effective_volume)
    print(
        f"MSE: {mse:.5e} "
        f"+/- {std_error:.5e}")
    print(
        f"Squared L2 error: {l2_error:.5e} "
        f"+/- {l2_std_error:.5e}")

    return


if __name__ == '__main__':
    main()
