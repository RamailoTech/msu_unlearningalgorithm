import os
import shutil
import pytest
import yaml

# Load configuration from YAML file
with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

@pytest.fixture
def setup_output_dir_adv_unlearn():
    output_dir = config.get("adv_unlearn", {}).get("output_dir", "results/adv_unlearn")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

# def test_adv_unlearn_run_compvis(setup_output_dir_adv_unlearn):
#     from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
#     from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
#     from mu.algorithms.erase_diff.configs import erase_diff_train_mu

#     try:
#         mu_defense = AdvUnlearnAlgorithm(
#             config=adv_unlearn_config,
#             compvis_ckpt_path = config['mu_defense']['compvis_ckpt_path'],  # path to the finetuned model
#             attack_step = config['mu_defense']['attack_step'],
#             backend = "compvis",
#             attack_method = config['mu_defense']['attack_method'],
#             train_method = config['mu_defense']['train_method'],  # training method; see docs for available options
#             warmup_iter = config['mu_defense']['warmup_iter'],
#             iterations = config['mu_defense']['iterations'],
#             model_config_path=erase_diff_train_mu.model_config_path  # use the same model config path for the used model
#         )
#         mu_defense.run()
#     except Exception as e:
#         pytest.fail(f"AdvUnlearnAlgorithm raised an exception: {str(e)}")

#     output_dir = os.path.join(adv_unlearn_config.output_dir,"models")
#     files = os.listdir(output_dir)
#     pt_files = [file for file in files if file.endswith('.pt')]
#     assert pt_files, f"No .pt files were found in the output directory {output_dir}. Files present: {files}"


# def test_adv_unlearn_run_diffusers(setup_output_dir_adv_unlearn):
#     import os
#     from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
#     from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config

#     try:
#         mu_defense = AdvUnlearnAlgorithm(
#             config = adv_unlearn_config,
#             diffusers_model_name_or_path = config['mu_defense']['diffusers_model_name_or_path'],  # path to the finetuned model
#             attack_step = config['mu_defense']['attack_step'],
#             backend = "diffusers",
#             attack_method = config['mu_defense']['attack_method'],
#             train_method = config['mu_defense']['train_method'],  # training method; see docs for available options
#             warmup_iter = config['mu_defense']['warmup_iter'],
#             iterations = config['mu_defense']['iterations'],
#         )
#         mu_defense.run()
#     except Exception as e:
#         pytest.fail(f"AdvUnlearnAlgorithm raised an exception: {str(e)}")


#     output_dir = os.path.join(adv_unlearn_config.output_dir,"models")
#     files = os.listdir(output_dir)
#     pt_files = [file for file in files if file.endswith('.pt')]
#     assert pt_files, f"No .pt files were found in the output directory {output_dir}. Files present: {files}"


def test_adv_unlearn_run_diffusers_without_text_encoder(setup_output_dir_adv_unlearn):
    import os
    from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
    from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config

    try:
        mu_defense = AdvUnlearnAlgorithm(
            config = adv_unlearn_config,
            diffusers_model_name_or_path = config['mu_defense']['diffusers_model_name_or_path'],  # path to the finetuned model
            attack_step = config['mu_defense']['attack_step'],
            backend = "diffusers",
            attack_method = config['mu_defense']['attack_method'],
            train_method = config['mu_defense']['train_method'],  # training method; see docs for available options
            warmup_iter = config['mu_defense']['warmup_iter'],
            iterations = config['mu_defense']['iterations'],
        )
        mu_defense.run()
    except Exception as e:
        pytest.fail(f"AdvUnlearnAlgorithm raised an exception: {str(e)}")


    expected_folders = [
        "feature_extractor",
        "logs",
        "scheduler",
        "text_encider",
        "tokenizer",
        "unet",
        "vae"
    ]
    expected_file = "model_index.json"

    output_dir = adv_unlearn_config.output_dir

    for folder in expected_folders:
        folder_path = os.path.join(output_dir, folder)
        assert os.path.isdir(folder_path), f"Expected folder '{folder}' not found in {output_dir}."

    # Check for expected file
    file_path = os.path.join(output_dir, expected_file)
    assert os.path.isfile(file_path), f"Expected file '{expected_file}' not found in {output_dir}."