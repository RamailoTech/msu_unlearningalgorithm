import os
import shutil
import pytest
import yaml

# Load configuration from YAML file
with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Common configuration shorthand (if needed)
common_config_mu = config["common_config_mu"]

@pytest.fixture
def setup_output_dir_muattack():
    output_dir = config['attack']['output_dir_diffuser']['output_dir']
    # Remove the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    # Cleanup after test
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def test_hard_prompt_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
    }

    try:
        MUAttack(
            config=hard_prompt_esd_nudity_P4D_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_hard_prompt_attack_run_compvis_to_diffuser_conversion(setup_output_dir_muattack):
    from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
        "task.save_diffuser": True,
        "task.sld": None,
        "task.model_name": config['attack']['hyperparameter']['model_name'],
    }

    try:
        MUAttack(
            config=hard_prompt_esd_nudity_P4D_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir'] 
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."

def test_no_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "attacker.no_attack.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
    }

    try:
        MUAttack(
            config=no_attack_esd_nudity_classifier_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_no_attack_run_compvis_to_diffuser_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "attacker.no_attack.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
        "task.save_diffuser": True,
        "task.sld": None,
        "task.model_name": config['attack']['hyperparameter']['model_name'],
    }

    try:
        MUAttack(
            config=no_attack_esd_nudity_classifier_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_random_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import random_esd_nudity_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
    }

    try:
        MUAttack(
            config=random_esd_nudity_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir'] 
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_random_attack_run_compvis_to_diffuser_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import random_esd_nudity_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
        "task.save_diffuser": True,
        "task.sld": None,
        "task.model_name": config['attack']['hyperparameter']['model_name'],
    }

    try:
        MUAttack(
            config=random_esd_nudity_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir'] 
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_text_grad_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
        "attacker.text_grad.lr": 0.02,
    }

    try:
        MUAttack(
            config=text_grad_esd_nudity_classifier_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir'] 
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_text_grad_attack_run_compvis_to_diffuser_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
        "attacker.text_grad.lr": 0.02,
        "task.save_diffuser": True,
        "task.sld": None,
        "task.model_name": config['attack']['hyperparameter']['model_name'],
    }

    try:
        MUAttack(
            config=text_grad_esd_nudity_classifier_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir'] 
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_seed_search_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import seed_search_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
    }

    try:
        MUAttack(
            config=seed_search_esd_nudity_classifier_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir'] 
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_seed_search_attack_run_compvis_to_diffuser(setup_output_dir_muattack):
    from mu_attack.configs.nudity import seed_search_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['model_and_dataset_path']['iterations'],
        "task.save_diffuser": True,
        "task.sld": None,
        "task.model_name": config['attack']['hyperparameter']['model_name'],
    }

    try:
        MUAttack(
            config=seed_search_esd_nudity_classifier_compvis_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dirs_compvis']['output_dir'] 
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."



######## **** TEST FOR DIFFUSERS MODEL **** ###########

def test_hard_prompt_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_diffusers_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.diffusers_model_name_or_path": config['attack']['model_and_dataset_path']['diffusers_model_name_or_path'],  # Path to the finetuned checkpoint
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dir_diffuser']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
    }

    try:
        MUAttack(
            config=hard_prompt_esd_nudity_P4D_diffusers_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dir_diffuser']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_no_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_diffusers_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": config['attack']['model_and_dataset_path']['diffusers_model_name_or_path'],  # Path to the finetuned checkpoint
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dir_diffuser']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
        "attacker.no_attack.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
    }

    try:
        MUAttack(
            config = no_attack_esd_nudity_classifier_diffusers_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dir_diffuser']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_no_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_diffuser_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": config['attack']['model_and_dataset_path']['diffusers_model_name_or_path'],  # Path to the finetuned checkpoint
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dir_diffuser']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
        "attacker.text_grad.lr": 0.02,
    }

    try:
        MUAttack(
            config = text_grad_esd_nudity_classifier_diffuser_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dir_diffuser']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_random_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import random_esd_nudity_diffuser_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": config['attack']['model_and_dataset_path']['diffusers_model_name_or_path'],  # Path to the finetuned checkpoint
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dir_diffuser']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
    }

    try:
        MUAttack(
            config = random_esd_nudity_diffuser_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dir_diffuser']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."


def test_seed_search_attack_run_compvis(setup_output_dir_muattack):
    from mu_attack.configs.nudity import seed_search_esd_nudity_classifier_diffusers_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": config['attack']['model_and_dataset_path']['diffusers_model_name_or_path'],  # Path to the finetuned checkpoint
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dir_diffuser']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
    }

    try:
        MUAttack(
            config = seed_search_esd_nudity_classifier_diffusers_config,
            **overridable_params
        )
    except Exception as e:
        pytest.fail(f"MUAttack raised an exception: {str(e)}")

    # Verify that the expected output directory was created.
    output_dir = config['attack']['output_dir_diffuser']['output_dir']
    assert os.path.exists(output_dir), f"Expected output directory {output_dir} was not created."

    # check for a log and config file or any other expected output.
    log_file = os.path.join(output_dir, "log.json")
    if os.path.exists(log_file):
        assert os.path.isfile(log_file), f"{log_file} is not a file."
    
    config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_file):
        assert os.path.isfile(config_file), f"{config_file} is not a file."