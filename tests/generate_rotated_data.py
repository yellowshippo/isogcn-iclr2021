import glob
import pathlib
import shutil
import subprocess

import femio
import numpy as np


def main():
    original_data_paths = [
        pathlib.Path(cnt_file).parent for cnt_file
        in glob.glob(
            'tests/data/simple/rotated_thermal/raw/**/thermal.cnt',
            recursive=True)]

    for original_data_path in original_data_paths:
        transform_data(original_data_path, additional_trial=2)
    return


def transform_data(data_path, *, additional_trial=2):
    fem_data = femio.FEMData.read_directory('fistr', data_path, read_npy=False)

    orthogonal_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    output_directory = data_path.parent / (data_path.name + 'rotated_x')
    process(fem_data, orthogonal_matrix, output_directory)

    for i in range(additional_trial):
        orthogonal_matrix = generate_rotation_matrix()
        output_directory = data_path.parent / (data_path.name + f"rotated_{i}")
        process(fem_data, orthogonal_matrix, output_directory)
    return


def process(fem_data, rotation_matrix, output_directory):
    _rotate_data(fem_data, rotation_matrix, output_directory)
    sp = subprocess.run(
        f"cd {output_directory} && fistr1", shell=True, check=True)
    print(sp)
    validate_results(fem_data, output_directory, rotation_matrix)
    np.savetxt(output_directory / 'rotation_matrix.txt', rotation_matrix)
    return


def generate_rotation_matrix():
    vec1 = normalize(np.random.rand(3)*2 - 1)
    vec2 = normalize(np.random.rand(3)*2 - 1)
    vec3 = normalize(np.cross(vec1, vec2))
    vec2 = np.cross(vec3, vec1)
    return np.array([vec1, vec2, vec3])


def normalize(x):
    return x / np.linalg.norm(x)


def validate_results(original_fem_data, output_directory, rotation_matrix):
    calculated_fem_data = femio.FEMData.read_directory(
        'fistr', output_directory, read_npy=False)
    rotated_original_strain = rotate_tensor_array(
        original_fem_data,
        original_fem_data.elemental_data.get_attribute_data(
            'ElementalSTRAIN'), rotation_matrix)
    calculated_strain = calculated_fem_data.elemental_data.get_attribute_data(
        'ElementalSTRAIN')
    mean_rmse = np.mean((
        rotated_original_strain - calculated_strain)**2)**.5
    ref = np.mean(rotated_original_strain**2)**.5
    print('========================')
    print(f"mean error: {mean_rmse}")
    print(f"relative mean error: {mean_rmse / ref * 100}")
    print('========================')
    with open(output_directory / 'log.txt', 'w') as f:
        f.write(f"mean error: {mean_rmse}\n")
        f.write(f"relative mean error: {mean_rmse / ref * 100}\n")
    if mean_rmse / ref * 100 > 1e-5:
        raise ValueError('Error too big')
    return


def rotate_tensor_array(fem_data, tensor_array, rotation_matrix):
    symmetric_mat = fem_data.convert_array2symmetric_matrix(
        tensor_array, from_engineering=True)
    rotated_mat = np.array([
        rotation_matrix @ l @ rotation_matrix.T for l in symmetric_mat])
    rotated_array = fem_data.convert_symmetric_matrix2array(
        rotated_mat, to_engineering=True)
    return rotated_array


def _rotate_data(fem_data, rotation_matrix, output_directory):
    shutil.rmtree(output_directory, ignore_errors=True)
    new_fem_data = femio.FEMData(
        fem_data.nodes, fem_data.elements)

    # Nodal data
    original_node = fem_data.nodal_data.get_attribute_data('NODE')
    rotated_nodes = np.array([
        rotation_matrix @ n for n in original_node])
    new_fem_data.nodes.data = rotated_nodes
    original_t_init = fem_data.nodal_data.get_attribute_data(
        'INITIAL_TEMPERATURE')
    original_t_cnt = fem_data.nodal_data.get_attribute_data('CNT_TEMPERATURE')
    nodal_data_dict = {
        'NODE': rotated_nodes,
        'INITIAL_TEMPERATURE': original_t_init,
        'CNT_TEMPERATURE': original_t_cnt}
    new_fem_data.nodal_data.update_data(
        new_fem_data.nodes.ids, nodal_data_dict, allow_overwrite=True)

    # Material data
    original_poisson_ratio = np.mean(
        fem_data.elemental_data.get_attribute_data('Poisson_ratio'),
        axis=0, keepdims=True)
    original_young_modulus = np.mean(
        fem_data.elemental_data.get_attribute_data('Young_modulus'),
        axis=0, keepdims=True)
    original_ltec = np.mean(
        fem_data.elemental_data.get_attribute_data(
            'linear_thermal_expansion_coefficient_full'),
        axis=0, keepdims=True)
    rotated_lte_array = rotate_tensor_array(
        new_fem_data, original_ltec, rotation_matrix)
    material_data_dict = {
        'Poisson_ratio': original_poisson_ratio,
        'Young_modulus': original_young_modulus,
        'linear_thermal_expansion_coefficient_full': rotated_lte_array}
    new_fem_data.materials.update_data('MAT_ALL', material_data_dict)

    # Elemental data
    n_element = len(new_fem_data.elements.ids)
    elemental_data_dict = {
        'Poisson_ratio': original_poisson_ratio * np.ones((n_element, 1)),
        'Young_modulus': original_young_modulus * np.ones((n_element, 1)),
        'linear_thermal_expansion_coefficient_full':
        rotated_lte_array * np.ones((n_element, 6))}
    new_fem_data.elemental_data.update_data(
        new_fem_data.elements.ids, elemental_data_dict)

    # Other info
    new_fem_data.settings = {
        'solution_type': 'STATIC',
        'output_res': 'NSTRAIN,ON\nNSTRESS,ON\n',
        'output_vis': 'NSTRAIN,ON\nNSTRESS,ON\n'}
    new_fem_data.element_groups = {'ALL': fem_data.elements.ids}
    new_fem_data.sections.update_data(
        'MAT_ALL', {'TYPE': 'SOLID', 'EGRP': 'ALL'})
    new_fem_data.constraints['spring'] = fem_data.constraints['spring']
    new_fem_data.material_overwritten = False

    new_fem_data.write('fistr', output_directory / 'mesh')
    return


if __name__ == '__main__':
    main()
