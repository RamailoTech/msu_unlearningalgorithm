import pytest
import yaml
import os
import shutil

# Load configuration from YAML file
with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Common config shorthand
common_config = config["common_config"]

# Fixture for erase_diff
@pytest.fixture
def setup_output_dir_erase_diff():
    output_dir = config['erase_diff']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

# Fixture for esd
@pytest.fixture
def setup_output_dir_esd():
    output_dir = config['esd']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

# Fixture for concept_ablation
@pytest.fixture
def setup_output_dir_concept_ablation():
    output_dir = config['concept_ablation']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def test_run_erase_diff(setup_output_dir_erase_diff):
    from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    algorithm = EraseDiffAlgorithm(
        erase_diff_train_mu,
        ckpt_path=common_config["model_dir"],
        raw_dataset_dir=common_config["input_data_dir"],
        template_name=common_config["template_name"],
        output_dir=config['erase_diff']['output_dir']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_erase_diff raised an exception: {str(e)}")

    output_dir = config['erase_diff']['output_dir']
    template_name = common_config["template_name"]
    output_filename = f"erase_diff_{template_name}_model.pth"
    expected_output_file = os.path.join(output_dir, output_filename)
    assert os.path.exists(expected_output_file), (
        f"Expected output file {expected_output_file} was not created"
    )
    assert os.path.isfile(expected_output_file), (
        f"{expected_output_file} is not a file"
    )
    assert expected_output_file.endswith('.pth'), (
        "Output file does not have .pth extension"
    )

def test_run_esd(setup_output_dir_esd):
    from mu.algorithms.esd.algorithm import ESDAlgorithm
    from mu.algorithms.esd.configs import esd_train_mu

    algorithm = ESDAlgorithm(
        esd_train_mu,
        ckpt_path=common_config["model_dir"],
        raw_dataset_dir=common_config["input_data_dir"],
        template_name=common_config["template_name"],
        output_dir=config['esd']['output_dir']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_esd raised an exception: {str(e)}")

    output_dir = config['esd']['output_dir']
    template_name = common_config["template_name"]
    output_filename = f"esd_{template_name}_model.pth"
    expected_output_file = os.path.join(output_dir, output_filename)
    assert os.path.exists(expected_output_file), (
        f"Expected output file {expected_output_file} was not created"
    )
    assert os.path.isfile(expected_output_file), (
        f"{expected_output_file} is not a file"
    )
    assert expected_output_file.endswith('.pth'), (
        "Output file does not have .pth extension"
    )

def test_run_concept_ablation(setup_output_dir_concept_ablation):
    from mu.algorithms.concept_ablation.algorithm import ConceptAblationAlgorithm
    from mu.algorithms.concept_ablation.configs import concept_ablation_train_mu

    concept_ablation_train_mu.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_mu,
        ckpt_path=common_config["model_dir"],
        prompts=config['concept_ablation']['prompts'],
        output_dir=config["concept_ablation"]["output_dir"],
        raw_dataset_dir=common_config["input_data_dir"]
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_concept_ablation raised an exception: {str(e)}")

    output_dir = config["concept_ablation"]["output_dir"]
    output_filename = "checkpoints/last.ckpt"  # Matches the actual output structure
    expected_output_file = os.path.join(output_dir, output_filename)
    assert os.path.exists(expected_output_file), (
        f"Expected output file {expected_output_file} was not created"
    )
    assert os.path.isfile(expected_output_file), (
        f"{expected_output_file} is not a file"
    )
    assert expected_output_file.endswith('.ckpt'), (
        "Output file does not have .ckpt extension"
    )

if __name__ == "__main__":
    pytest.main([__file__])