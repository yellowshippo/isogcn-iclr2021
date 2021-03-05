import argparse
import glob
import multiprocessing as multi
import pathlib
from distutils.util import strtobool

import numpy as np
import scipy.sparse as sp
import siml
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory',
        type=pathlib.Path,
        help='Input data directory path')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, overwrite existing data [False]')
    args = parser.parse_args()

    if not args.data_directory.is_dir():
        raise ValueError(f"Directory does not exist: {args.data_directory}")

    data_directories = [
        pathlib.Path(f).parent for f in glob.glob(
            str(args.data_directory / '**/node.npy'),
            recursive=True)]
    max_process = siml.util.determine_max_process()
    chunksize = max(
        len(data_directories) // max_process // 16, 1)

    with multi.Pool(max_process) as pool:
        grad_stats = pool.map(
            calculate_grad_stats, data_directories, chunksize=chunksize)
    dict_scales = summarize_scales(grad_stats)
    print(dict_scales)

    output_file = args.data_directory / 'grad_stats.yml'
    if not args.force_renew and output_file.exists():
        raise ValueError(f"File already exists: {output_file}")
    with open(output_file, 'w') as f:
        yaml.dump(dict_scales, f)
    return


def summarize_scales(grad_stats):
    grad_keys = grad_stats[0].keys()
    dict_n = {
        k: np.sum(np.fromiter(
            (grad_stat[k][0] for grad_stat in grad_stats), int))
        for k in grad_keys}
    dict_sum = {
        k: np.sum(np.fromiter(
            (grad_stat[k][1] for grad_stat in grad_stats), float))
        for k in grad_keys}
    dict_scales = {k: 1 / float(dict_sum[k] / dict_n[k])**.5 for k in grad_keys}
    return dict_scales


def calculate_grad_stats(input_directory):
    print(f"Processing: {input_directory}")
    grad_x_files = glob.glob(str(input_directory / '*grad_x*.npz'))
    return {
        pathlib.Path(grad_x_file).stem.replace('_x', ''):
        _calculate_stat(pathlib.Path(grad_x_file))
        for grad_x_file in grad_x_files}


def _calculate_stat(grad_x_file):
    grad_x = sp.load_npz(grad_x_file)
    parent = grad_x_file.parent
    grad_y = sp.load_npz(parent / grad_x_file.name.replace('x', 'y'))
    grad_z = sp.load_npz(parent / grad_x_file.name.replace('x', 'z'))
    n = grad_x.shape[0]
    scale = np.sum(
        grad_x.diagonal()**2 + grad_y.diagonal()**2 + grad_z.diagonal()**2)
    return n, scale


if __name__ == '__main__':
    main()
