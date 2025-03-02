import os
import shutil
import pytest
import yaml

# Load configuration from YAML file
with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)


common_config_unlearn_canvas = config["common_config_unlearn_canvas_mu"]
output_dir = config['erase_diff']['output_dir']
template_name = common_config_unlearn_canvas["template_name"]
output_filename = f"erase_diff_{template_name}_model.pth"
compvis_model_checkpoint = os.path.join(output_dir, output_filename)
forget_me_not_model_output_dir = config['forget_me_not']['finetuned_output_dir']

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

@pytest.fixture(scope="session")
def run_erase_diff():
    from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    algorithm = EraseDiffAlgorithm(
        erase_diff_train_mu,
        ckpt_path=common_config_unlearn_canvas["compvis_model_dir"],
        raw_dataset_dir=common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        template_name=common_config_unlearn_canvas["template_name"],
        output_dir=config['erase_diff']['output_dir'],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type']
    )
    
    algorithm.run()

    output_dir = config['erase_diff']['output_dir']
    template_name = common_config_unlearn_canvas["template_name"]
    output_filename = f"erase_diff_{template_name}_model.pth"
    expected_output_file = os.path.join(output_dir, output_filename)
    return expected_output_file

def test_hard_prompt_attack_run_compvis(run_erase_diff):
    from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_no_attack_run_compvis(run_erase_diff):
    from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "attacker.no_attack.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_evaluator():
    from mu_attack.configs.evaluation import attack_evaluation_config
    from mu_attack.execs.evaluator import MuAttackEvaluator
    mu_attack_config = config['attack']
    eval_config = attack_evaluation_config
    eval_config.asr.root = f"{mu_attack_config['output_dirs_compvis']['output_dir']}/P4d"
    eval_config.asr.root_no_attack = f"{mu_attack_config['output_dirs_compvis']['output_dir']}/NoAttackEsdNudity"
    eval_config.clip.devices = "0"
    eval_config.clip.image_path = f"{mu_attack_config['output_dirs_compvis']['output_dir']}/P4d/images"
    eval_config.clip.log_path = f"{mu_attack_config['output_dirs_compvis']['output_dir']}/P4d/log.json"
    eval_config.fid.ref_batch_path = f"{mu_attack_config['output_dirs_compvis']['output_dir']}/NoAttackEsdNudity/images"  #to calculate fid score, the genrated image path and ref image path should have ssame number of images.
    eval_config.fid.sample_batch_path = mu_attack_config['evaluation']['sample_batch_path'] #should have same number of images as of ref batch.
    eval_config.output_path = "testing/evaluation/results.json"

    evaluator = MuAttackEvaluator(eval_config)
    try:
        evaluator.run()
    except Exception as e:
        pytest.fail(f"MUAttack evalaution raised an exception: {str(e)}")

def test_hard_prompt_attack_run_compvis_to_diffuser_conversion(setup_output_dir_muattack,run_erase_diff):
    from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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



def test_no_attack_run_compvis_to_diffuser_compvis(setup_output_dir_muattack, run_erase_diff):
    from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "attacker.no_attack.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_random_attack_run_compvis(setup_output_dir_muattack, run_erase_diff):
    from mu_attack.configs.nudity import random_esd_nudity_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_random_attack_run_compvis_to_diffuser_compvis(setup_output_dir_muattack, run_erase_diff):
    from mu_attack.configs.nudity import random_esd_nudity_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_text_grad_attack_run_compvis(setup_output_dir_muattack, run_erase_diff):
    from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": config['attack']['model_and_dataset_path']['compvis_model_and_dataset_path'],  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_text_grad_attack_run_compvis_to_diffuser_compvis(setup_output_dir_muattack, run_erase_diff):
    from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_seed_search_attack_run_compvis(setup_output_dir_muattack, run_erase_diff):
    from mu_attack.configs.nudity import seed_search_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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


def test_seed_search_attack_run_compvis_to_diffuser(setup_output_dir_muattack, run_erase_diff):
    from mu_attack.configs.nudity import seed_search_esd_nudity_classifier_compvis_config
    from mu_attack.execs.attack import MUAttack
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    overridable_params = {
        "task.compvis_ckpt_path": compvis_model_checkpoint,  # Path to the finetuned checkpoint
        "task.compvis_config_path": erase_diff_train_mu.model_config_path,  # CompVis model configuration path
        "task.dataset_path": config['attack']['model_and_dataset_path']['dataset_path'],
        "logger.json.root": config['attack']['output_dirs_compvis']['output_dir'] ,
        "attacker.iteration": config['attack']['hyperparameter']['iterations'],
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



# ######## **** TEST FOR DIFFUSERS MODEL **** ###########


@pytest.fixture(scope="session")
def forget_me_not_model_output():
    from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
    from mu.algorithms.forget_me_not.configs import forget_me_not_train_ti_mu, forget_me_not_train_attn_mu

    ti_output_dir = config['forget_me_not']['ti_output_dir']
    if not os.path.exists(ti_output_dir):
        os.makedirs(ti_output_dir)
    
    steps = config['forget_me_not']['steps']
    

    algorithm_ti = ForgetMeNotAlgorithm(
         forget_me_not_train_ti_mu,
         ckpt_path = common_config_unlearn_canvas["diffuser_model_dir"],
         raw_dataset_dir = common_config_unlearn_canvas['unlearn_canvas_data_dir'],
         steps = steps,
         use_sample = common_config_unlearn_canvas['use_sample'],
         dataset_type = common_config_unlearn_canvas['dataset_type'],
         template_name = common_config_unlearn_canvas["template_name"],
         template = common_config_unlearn_canvas['template'],
         output_dir = ti_output_dir
    )
    algorithm_ti.run(train_type="train_ti")

    ti_weights_path = os.path.join(ti_output_dir, f"step_inv_{steps}.safetensors")
    
    algorithm_attn = ForgetMeNotAlgorithm(
         forget_me_not_train_attn_mu,
         ckpt_path = common_config_unlearn_canvas["diffuser_model_dir"],
         raw_dataset_dir = common_config_unlearn_canvas['unlearn_canvas_data_dir'],
         steps = 10, 
         ti_weights_path = ti_weights_path,
         devices = config['forget_me_not']['devices'],
         template = common_config_unlearn_canvas['template'],
         output_dir = config['forget_me_not']['finetuned_output_dir']
    )
    algorithm_attn.run(train_type="train_attn")

    # The expected finetuned model directory is defined in the config.
    output_dir = config['forget_me_not']['finetuned_output_dir']
    template_name = common_config_unlearn_canvas["template_name"]
    expected_model_dir = os.path.join(output_dir, template_name)

    # Return the finetuned model directory so that it can be used as the input for AdvUnlearn.
    return expected_model_dir


def test_hard_prompt_attack_run_diffuser(setup_output_dir_muattack, forget_me_not_model_output):
    from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_diffusers_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": forget_me_not_model_output_dir,  # Path to the finetuned checkpoint
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


def test_no_attack_run_diffuser(setup_output_dir_muattack, forget_me_not_model_output):
    from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_diffusers_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": forget_me_not_model_output_dir,  # Path to the finetuned checkpoint
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


def test_no_attack_run_compvis(setup_output_dir_muattack,forget_me_not_model_output):
    from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_diffuser_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": forget_me_not_model_output_dir,  # Path to the finetuned checkpoint
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


def test_random_attack_run_diffuser(setup_output_dir_muattack, forget_me_not_model_output):
    from mu_attack.configs.nudity import random_esd_nudity_diffuser_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": forget_me_not_model_output_dir,  # Path to the finetuned checkpoint
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


def test_seed_search_attack_run_diffuser(setup_output_dir_muattack, forget_me_not_model_output):
    from mu_attack.configs.nudity import seed_search_esd_nudity_classifier_diffusers_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.diffusers_model_name_or_path": forget_me_not_model_output_dir,  # Path to the finetuned checkpoint
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