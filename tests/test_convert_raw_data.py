from pathlib import Path
import shutil
import sys
import unittest

import numpy as np

import siml.inferer as inferer
import siml.prepost as pre
import siml.setting as setting
import siml.trainer as trainer

sys.path.insert(0, 'src')
import convert_raw_data as convert  # NOQA


class TestConvertRowData(unittest.TestCase):

    def test_grad_grad_vs_laplacian(self):
        preprocess_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/precalculated/convert.yml'))
        shutil.rmtree(preprocess_setting.data.interim, ignore_errors=True)
        shutil.rmtree(preprocess_setting.data.preprocessed, ignore_errors=True)

        raw_converter = pre.RawConverter(
            preprocess_setting,
            conversion_function=convert.HeatConversionFuncionCreator(
                create_elemental=False),
            filter_function=convert.filter_function_heat,
            force_renew=True, recursive=True, to_first_order=True,
            write_ucd=False, read_npy=False, read_res=True)
        raw_converter.convert()
        preprocessor = pre.Preprocessor(
            preprocess_setting, force_renew=True)
        preprocessor.preprocess_interim_data()

        # Confirm results does not change under rigid body transformation
        grad_grad_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/precalculated/grad_grad.yml'))
        grad_grad_tr = trainer.Trainer(grad_grad_setting)
        if grad_grad_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(grad_grad_tr.setting.trainer.output_directory)
        grad_grad_tr.train()

        grad_grad_ir = inferer.Inferer(grad_grad_setting)
        inference_outpout_directory = \
            grad_grad_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = grad_grad_ir.infer(
            model=Path('tests/data/precalculated/models/grad_grad'),
            preprocessed_data_directory=[
                Path(
                    'tests/data/precalculated/preprocessed/cube/clscale1.0/'
                    'original'),
                Path(
                    'tests/data/precalculated/preprocessed/cube/clscale1.0/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/precalculated/preprocessed/preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True)
        np.testing.assert_almost_equal(
            results[0]['dict_y']['nodal_t_100'],
            results[1]['dict_y']['nodal_t_100'])

        laplacian_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/precalculated/laplacian.yml'))
        laplacian_tr = trainer.Trainer(laplacian_setting)
        if laplacian_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(laplacian_tr.setting.trainer.output_directory)
        laplacian_tr.train()

        laplacian_ir = inferer.Inferer(laplacian_setting)
        inference_outpout_directory = \
            laplacian_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = laplacian_ir.infer(
            model=Path('tests/data/precalculated/models/laplacian'),
            preprocessed_data_directory=[
                Path(
                    'tests/data/precalculated/preprocessed/cube/clscale1.0/'
                    'original'),
                Path(
                    'tests/data/precalculated/preprocessed/cube/clscale1.0/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/precalculated/preprocessed/preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True)
        np.testing.assert_almost_equal(
            results[0]['dict_y']['nodal_t_100'],
            results[1]['dict_y']['nodal_t_100'])

    def test_raw_converter_skip_unconvergence(self):
        conversion_function = convert.ThermalConversionFuncionCreator(
            create_elemental=False, read_npy=False, thermal_mode='tensor')
        raw_converter = pre.RawConverter.read_settings(
            Path('tests/data/check_convergence/data.yml'),
            conversion_function=conversion_function,
            filter_function=convert.ThermalFilterFunctionCreator(1.e-2),
            force_renew=True, recursive=True, to_first_order=True,
            write_ucd=False, read_npy=False, read_res=True)
        raw_converter.convert()
