import argparse
from distutils.util import strtobool
import pathlib

import siml


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
    args = parser.parse_args()

    preprocessor = siml.prepost.Preprocessor.read_settings(
        args.settings_yaml, force_renew=args.force_renew)
    preprocessor.preprocess_interim_data()
    # preprocessor.preprocess_interim_data(
    #     force_renew=args.force_renew, save_func=save_func)

    print('success')


def save_func(output_directory, variable_name, transformed_data):
    n_step = 3
    for i_step in range(1, n_step):
        step_output_directory = output_directory / f"step{i_step}"
        step_output_directory.mkdir(parents=True, exist_ok=True)
        if variable_name in ['temperature', 'ave_temperature']:
            if variable_name == 'temperature':
                root_name = 't'
            elif variable_name == 'ave_temperature':
                root_name = 'ave_t'
            else:
                raise ValueError

            siml.util.save_variable(
                step_output_directory, root_name + '_pre',
                transformed_data[i_step - 1])
            siml.util.save_variable(
                step_output_directory, root_name + '_post',
                transformed_data[i_step])
        else:
            siml.util.save_variable(
                step_output_directory, variable_name, transformed_data)
        (
            step_output_directory
            / siml.prepost.Preprocessor.FINISHED_FILE).touch()
    return


if __name__ == '__main__':
    main()
