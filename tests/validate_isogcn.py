import argparse
from distutils.util import strtobool
import glob
import pathlib
import sys

import siml

sys.path.append('.')
from lib.siml.tests import test_iso_gcn  # NOQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        'original_root_directory',
        type=pathlib.Path,
        help='Root directory which contains data')
    parser.add_argument(
        '-s', '--rank0-variable-name',
        type=str,
        default=None,
        help='Variable name of rank 0 tensor data')
    parser.add_argument(
        '-t', '--rank2-variable-name',
        type=str,
        default=None,
        help='Variable name of rank 2 tensor data')
    parser.add_argument(
        '-e', '--threshold_percent',
        type=float,
        default=.1,
        help='Threshold for relative comparison')
    parser.add_argument(
        '-d', '--decimal',
        type=int,
        default=3,
        help='Decimal for numpy testing.')
    parser.add_argument(
        '-o', '--output-directory',
        type=pathlib.Path,
        default=None,
        help='Output directory name. '
        'If not fed, will be determined automatically.')
    parser.add_argument(
        '-w', '--write-ucd',
        type=strtobool,
        default=0,
        help='If True, write AVS UCD file of converted interim data [False]')
    parser.add_argument(
        '-r', '--reference-name',
        type=str,
        default='thermal_rep[0-9]',
        help='Reference directory name')
    args = parser.parse_args()

    if args.rank0_variable_name is None and args.rank2_variable_name is None:
        raise ValueError('Feed at least one rank0 or rank2 variable name')

    original_data_directories = [
        pathlib.Path(d) for d
        in glob.glob(
            str(args.original_root_directory / ('**/' + args.reference_name)),
            recursive=True)]
    map_data_directories = [
        {
            'original': original_data_directory,
            'list_transformed': [
                pathlib.Path(d) for d in glob.glob(
                    str(original_data_directory) + '_transformed_*')]}
        for original_data_directory in original_data_directories]
    validator = IsoGCNValudator(map_data_directories, args)
    validator.validate()
    return


class IsoGCNValudator(test_iso_gcn.TestIsoGCN):

    def __init__(self, map_data_directories, settings):
        self.map_data_directories = map_data_directories
        self.settings = settings
        self._initialize_inferer()
        return

    def validate(self):
        for map_data_directory in self.map_data_directories:
            original_results = self.infer(map_data_directory['original'])
            transformed_results = self.infer(
                map_data_directory['list_transformed'])
            self.validate_results(
                original_results, transformed_results,
                rank0=self.settings.rank0_variable_name,
                rank2=self.settings.rank2_variable_name,
                decimal=self.settings.decimal,
                threshold_percent=self.settings.threshold_percent)
        return

    def infer(self, data_directory):
        if isinstance(data_directory, list):
            data_directories = data_directory
        else:
            data_directories = [data_directory]
        print(f"Model: {self.model_directory}")
        return self.inferer.infer(
            model=self.model_directory,
            preprocessed_data_directory=data_directories,
            converter_parameters_pkl=self.settings.original_root_directory
            / 'preprocessors.pkl', overwrite=True, perform_postprocess=False)

    def _initialize_inferer(self):
        main_setting = siml.setting.MainSetting.read_settings_yaml(
            self.settings.settings_yaml)
        main_setting.data.train = [self.map_data_directories[0]['original']]
        main_setting.data.validation = [
            self.map_data_directories[0]['original']]
        main_setting.data.test = []
        main_setting.data.develop = []
        if self.settings.output_directory is None:
            main_setting.trainer.update_output_directory()
        else:
            main_setting.trainer.output_directory \
                = self.settings.output_directory
        main_setting.trainer.split_ratio = {}
        main_setting.trainer.n_epoch = 3
        main_setting.trainer.log_trigger_epoch = 1
        main_setting.trainer.stop_trigger_epoch = 1
        main_setting.trainer.batch_size = 1
        main_setting.trainer.validation_batch_size = 1

        trainer = siml.trainer.Trainer(main_setting)
        trainer.train()

        self.inferer = siml.inferer.Inferer(main_setting)
        self.model_directory = str(main_setting.trainer.output_directory)
        return


if __name__ == '__main__':
    main()
