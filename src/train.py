import argparse
from distutils.util import strtobool
import pathlib

import siml
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=None,
        help='If fed, sed batch size')
    parser.add_argument(
        '-d', '--data-parallel',
        type=strtobool,
        default=0,
        help='If True, perform data parallelism [False]')
    parser.add_argument(
        '-g', '--gpu-id',
        type=int,
        default=-1,
        help='GPU ID [-1, meaning CPU]')
    parser.add_argument(
        '-r', '--restart-dir',
        type=pathlib.Path,
        default=None,
        help='Restart directory name')
    parser.add_argument(
        '-p', '--pretrained-directory',
        type=pathlib.Path,
        default=None,
        help='Pretrained directory name')
    parser.add_argument(
        '-n', '--n-epoch',
        type=int,
        default=None,
        help='If fed, set the number of epochs.')
    args = parser.parse_args()

    main_setting = siml.setting.MainSetting.read_settings_yaml(
        args.settings_yaml)
    if args.out_dir is None:
        args.out_dir = pathlib.Path(str(args.settings_yaml).replace(
            'inputs/', 'models/').replace('.yml', ''))

    main_setting.trainer.output_directory = args.out_dir
    main_setting.trainer.gpu_id = args.gpu_id
    if args.restart_dir is not None:
        main_setting.trainer.restart_directory = args.restart_dir
    if args.pretrained_directory is not None:
        main_setting.trainer.pretrain_directory = args.pretrained_directory
    if args.batch_size is not None:
        main_setting.trainer.batch_size = args.batch_size
        main_setting.trainer.validation_batch_size = args.batch_size
    if args.n_epoch is not None:
        main_setting.trainer.n_epoch = args.n_epoch
    if args.data_parallel:
        main_setting.trainer.data_parallel = args.data_parallel
        gpu_count = torch.cuda.device_count()
        original_batch_size = main_setting.trainer.batch_size
        main_setting.trainer.batch_size = original_batch_size * gpu_count
        main_setting.trainer.validation_batch_size \
            = main_setting.trainer.batch_size
        main_setting.trainer.num_workers = main_setting.trainer.num_workers \
            * gpu_count
        print(f"Batch size: {original_batch_size} x {gpu_count} GPUs")

    trainer = siml.trainer.Trainer(main_setting)
    trainer.train()
    return


if __name__ == '__main__':
    main()
