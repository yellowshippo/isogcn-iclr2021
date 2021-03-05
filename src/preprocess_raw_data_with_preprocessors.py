import argparse
from distutils.util import strtobool
import pathlib

import siml

import convert_raw_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        'raw_data_directory',
        type=pathlib.Path,
        help='Raw data directory')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Preprocessors.pkl file')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, overwrite existing data [False]')
    parser.add_argument(
        '-l', '--light',
        type=strtobool,
        default=0,
        help='If True, compute minimum required data only [False]')
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
        '-a', '--convert-answer',
        type=strtobool,
        default=1,
        help='If True, convert answer [True]')
    parser.add_argument(
        '-s', '--skip-interim',
        type=strtobool,
        default=0,
        help='If True, skip conversion of interim data [False]')
    args = parser.parse_args()

    main_setting = siml.setting.MainSetting.read_settings_yaml(
        args.settings_yaml)
    if not args.convert_answer:
        main_setting.conversion.required_file_names = ['*.msh', '*.cnt']
    main_setting.data.raw = args.raw_data_directory
    if args.out_dir is None:
        args.out_dir = args.raw_data_directory
        main_setting.data.interim = [siml.prepost.determine_output_directory(
            main_setting.data.raw,
            main_setting.data.raw.parent / 'interim', 'raw')]
        main_setting.data.preprocessed = [
            siml.prepost.determine_output_directory(
                main_setting.data.raw,
                main_setting.data.raw.parent / 'preprocessed', 'raw')]
    else:
        main_setting.data.interim = [args.out_dir / 'interim']
        main_setting.data.preprocessed = [args.out_dir / 'preprocessed']

    if not args.skip_interim:
        conversion_function = convert_raw_data.HeatConversionFuncionCreator(
            create_elemental=args.elemental,
            convert_answer=args.convert_answer,
            light=args.light)
        raw_converter = siml.prepost.RawConverter(
            main_setting,
            conversion_function=conversion_function,
            filter_function=convert_raw_data.filter_function_heat,
            force_renew=args.force_renew,
            recursive=args.recursive,
            to_first_order=True,
            write_ucd=False,
            read_npy=args.read_npy, read_res=args.convert_answer)
        raw_converter.convert()

    preprocessor = siml.prepost.Preprocessor(
        main_setting, force_renew=args.force_renew,
        allow_missing=True)
    preprocessor.convert_interim_data(preprocessor_pkl=args.preprocessors_pkl)

    return


if __name__ == '__main__':
    main()
