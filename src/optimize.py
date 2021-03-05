import argparse
import pathlib

import siml


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
        '-g', '--gpu-id',
        type=int,
        default=-1,
        help='GPU ID [-1, meaning CPU]')
    parser.add_argument(
        '-d', '--db-settings-yaml',
        type=pathlib.Path,
        default=None,
        help='DB setting file [None, meaning the same as settings_yaml]')
    args = parser.parse_args()

    main_setting = siml.setting.MainSetting.read_settings_yaml(
        args.settings_yaml)
    if args.db_settings_yaml is None:
        db_setting = None
    else:
        db_setting = siml.setting.DBSetting.read_settings_yaml(
            args.db_settings_yaml)

    if args.out_dir is not None:
        main_setting.trainer.out_dir(args.out_dir)
    main_setting.trainer.gpu_id = args.gpu_id

    siml.optimize.perform_study(main_setting, db_setting)


if __name__ == '__main__':
    main()
