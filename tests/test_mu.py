import pytest
import yaml
import os
import shutil

# Load configuration from YAML file
with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

common_config_unlearn_canvas = config["common_config_unlearn_canvas_mu"]
common_config_i2p = config["common_config_i2p"]

# Fixture for erase_diff
@pytest.fixture
def setup_output_dir_erase_diff():
    output_dir = config['erase_diff']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
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

@pytest.fixture
def setup_output_dir_scissorhands():
    output_dir = config['scissorhands']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

@pytest.fixture
def setup_output_dir_unified_concept_editing():
    output_dir = config['unified_concept_editing']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

@pytest.fixture
def setup_output_dir_forget_me_not_ti():
    output_dir = config['forget_me_not']['ti_output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

@pytest.fixture
def setup_output_dir_forget_me_not_finetuned():
    output_dir = config['forget_me_not']['finetuned_output_dir']
    ti_output_dir = config['forget_me_not']['ti_output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
            
    if os.path.exists(ti_output_dir):
        shutil.rmtree(ti_output_dir)


@pytest.fixture
def setup_output_dir_saliency_unlearning():
    output_dir = config['saliency_unlearning']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


@pytest.fixture
def setup_output_dir_semipermeable():
    output_dir = config['semipermeable']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


@pytest.fixture
def setup_output_dir_selective_amnesia():
    output_dir = config['selective_amnesia']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def test_run_erase_diff(setup_output_dir_erase_diff):
    from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu, erase_diff_evaluation_config
    from mu.algorithms.erase_diff import EraseDiffEvaluator
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    algorithm = EraseDiffAlgorithm(
        erase_diff_train_mu,
        ckpt_path=common_config_unlearn_canvas["compvis_model_dir"],
        raw_dataset_dir=common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        template_name=common_config_unlearn_canvas["template_name"],
        output_dir=config['erase_diff']['output_dir'],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_erase_diff raised an exception: {str(e)}")

    output_dir = config['erase_diff']['output_dir']
    template_name = common_config_unlearn_canvas["template_name"]
    output_filename = f"erase_diff_{template_name}_model.pth"
    erase_diff_ckpt_path = os.path.join(output_dir, output_filename)
    assert os.path.exists(erase_diff_ckpt_path), (
        f"Expected output file {erase_diff_ckpt_path} was not created"
    )
    assert os.path.isfile(erase_diff_ckpt_path), (
        f"{erase_diff_ckpt_path} is not a file"
    )
    assert erase_diff_ckpt_path.endswith('.pth'), (
        "Output file does not have .pth extension"
    )

    # run the evaluator
    evaluator = EraseDiffEvaluator(
        erase_diff_evaluation_config,
        ckpt_path = erase_diff_ckpt_path,
    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"erase diff evaluator raised an exception: {str(e)}")


def test_run_esd(setup_output_dir_esd): 
    from mu.algorithms.esd.algorithm import ESDAlgorithm
    from mu.algorithms.esd.configs import esd_train_mu, esd_evaluation_config
    from mu.algorithms.esd import ESDEvaluator
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    algorithm = ESDAlgorithm(
        esd_train_mu,
        ckpt_path=common_config_unlearn_canvas["compvis_model_dir"],
        raw_dataset_dir=common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        template_name=common_config_unlearn_canvas["template_name"],
        output_dir=config['esd']['output_dir'],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template=common_config_unlearn_canvas['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_esd raised an exception: {str(e)}")

    output_dir = config['esd']['output_dir']
    template_name = common_config_unlearn_canvas["template_name"]
    output_filename = f"esd_{template_name}_model.pth"
    esd_output_ckpt_path = os.path.join(output_dir, output_filename)
    assert os.path.exists(esd_output_ckpt_path), (
        f"Expected output file {esd_output_ckpt_path} was not created"
    )
    assert os.path.isfile(esd_output_ckpt_path), (
        f"{esd_output_ckpt_path} is not a file"
    )
    assert esd_output_ckpt_path.endswith('.pth'), (
        "Output file does not have .pth extension"
    )

    # run the evaluator
    evaluator = ESDEvaluator(
        esd_evaluation_config,
        ckpt_path = esd_output_ckpt_path,

    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"esd evaluator raised an exception: {str(e)}")



def test_run_concept_ablation(setup_output_dir_concept_ablation):
    from mu.algorithms.concept_ablation.algorithm import ConceptAblationAlgorithm
    from mu.algorithms.concept_ablation import ConceptAblationEvaluator
    from mu.algorithms.concept_ablation.configs import concept_ablation_train_mu, concept_ablation_evaluation_config
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    concept_ablation_train_mu.lightning.trainer.max_steps = 1

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_mu,
        ckpt_path=common_config_unlearn_canvas["compvis_model_dir"],
        prompts=config['concept_ablation']['prompts'],
        output_dir=config["concept_ablation"]["output_dir"],
        raw_dataset_dir=common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_concept_ablation raised an exception: {str(e)}")

    output_dir = config["concept_ablation"]["output_dir"]
    output_filename = "checkpoints/last.ckpt"  # Matches the actual output structure
    concept_ablation_output_ckpt_path = os.path.join(output_dir, output_filename)
    assert os.path.exists(concept_ablation_output_ckpt_path), (
        f"Expected output file {concept_ablation_output_ckpt_path} was not created"
    )
    assert os.path.isfile(concept_ablation_output_ckpt_path), (
        f"{concept_ablation_output_ckpt_path} is not a file"
    )
    assert concept_ablation_output_ckpt_path.endswith('.ckpt'), (
        "Output file does not have .ckpt extension"
    )
    print("Checkpoint path:", concept_ablation_output_ckpt_path)
    # run the evaluator
    evaluator = ConceptAblationEvaluator(
        concept_ablation_evaluation_config,
        # ckpt_path = concept_ablation_output_ckpt_path,
        ckpt_path ="outputs/concept_ablation/checkpoints/last.ckpt"
    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"concept ablation evalautor raised an exception: {str(e)}")


def test_run_scissorhands(setup_output_dir_scissorhands):
    from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
    from mu.algorithms.scissorhands.configs import scissorhands_train_mu, scissorhands_evaluation_config
    from mu.algorithms.scissorhands import ScissorHandsEvaluator
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    algorithm = ScissorHandsAlgorithm(
        scissorhands_train_mu,
        ckpt_path=common_config_unlearn_canvas["compvis_model_dir"],
        raw_dataset_dir=common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        output_dir=config['scissorhands']['output_dir'],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_scissorhands raised an exception: {str(e)}")

    output_dir = config['scissorhands']['output_dir']
    template_name = common_config_unlearn_canvas["template_name"]
    output_filename = f"scissorhands_{template_name}_model.pth"
    scissorhands_output_ckpt_path = os.path.join(output_dir, output_filename)
    assert os.path.exists(scissorhands_output_ckpt_path), (
        f"Expected output file {scissorhands_output_ckpt_path} was not created"
    )
    assert os.path.isfile(scissorhands_output_ckpt_path), (
        f"{scissorhands_output_ckpt_path} is not a file"
    )
    assert scissorhands_output_ckpt_path.endswith('.pth'), (
        "Output file does not have .pth extension"
    )

    #run the evaluator
    evaluator = ScissorHandsEvaluator(
        scissorhands_evaluation_config,
        ckpt_path = scissorhands_output_ckpt_path,
    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"scissorhands evalautor raised an exception: {str(e)}")


def test_run_unified_concept_editing(setup_output_dir_unified_concept_editing):
    from mu.algorithms.unified_concept_editing.algorithm import UnifiedConceptEditingAlgorithm
    from mu.algorithms.unified_concept_editing import UnifiedConceptEditingEvaluator
    from mu.algorithms.unified_concept_editing.configs import unified_concept_editing_train_mu, uce_evaluation_config
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    # Use the provided checkpoint path for unified concept editing
    algorithm = UnifiedConceptEditingAlgorithm(
        unified_concept_editing_train_mu,
        ckpt_path = common_config_unlearn_canvas['diffuser_model_dir'],
        raw_dataset_dir=common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        output_dir=config['unified_concept_editing']['output_dir'],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_unified_concept_editing raised an exception: {str(e)}")

    output_dir = config['unified_concept_editing']['output_dir']
    template_name = common_config_unlearn_canvas["template_name"]
    uce_output_ckpt_path = os.path.join(output_dir, f"uce_{template_name}_model")
    
    # Verify that the main output directory was created
    assert os.path.exists(uce_output_ckpt_path), (
        f"Expected output directory {uce_output_ckpt_path} was not created"
    )
    

    expected_subdirs = [
        "feature_extractor", "safety_checker", "scheduler", "text_encoder",
        "tokenizer", "unet", "vae"
    ]
    for subdir in expected_subdirs:
        subdir_path = os.path.join(uce_output_ckpt_path, subdir)
        assert os.path.exists(subdir_path), (
            f"Expected subdirectory {subdir_path} was not created"
        )
    
    model_index = os.path.join(uce_output_ckpt_path, "model_index.json")
    assert os.path.exists(model_index), (
        f"Expected model index file {model_index} was not created"
    )

    #run the evaluator
    evaluator = UnifiedConceptEditingEvaluator(
        uce_evaluation_config,
        ckpt_path = uce_output_ckpt_path,

    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"Unified concept editing evalautor raised an exception: {str(e)}")

def test_generate_safe_tensors():
    """
    This test runs the generate_safe_tensors function from the ForgetMeNot algorithm
    and checks that the .safetensors file is created.
    """
    from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
    from mu.algorithms.forget_me_not.configs import forget_me_not_train_ti_mu

    algorithm = ForgetMeNotAlgorithm(
        forget_me_not_train_ti_mu,
        ckpt_path = common_config_unlearn_canvas["diffuser_model_dir"],
        raw_dataset_dir = common_config_unlearn_canvas['unlearn_canvas_data_dir'],
        steps = config['forget_me_not']['steps'],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template'],
        output_dir = config['forget_me_not']['ti_output_dir']
    )
    try:
        algorithm.run(train_type="train_ti")
    except Exception as e:
        pytest.fail(f"generate_safe_tensors raised an exception: {str(e)}")

    # Expected safe tensor file
    steps = config['forget_me_not']['steps']
    expected_file = os.path.join(config['forget_me_not']['ti_output_dir'], f"step_inv_{steps}.safetensors")
    assert os.path.exists(expected_file), (
        f"Expected safe tensor file {expected_file} was not created"
    )
    assert expected_file.endswith('.safetensors'), (
        "Output file does not have .safetensors extension"
    )

def test_run_mu_forget_me_not(setup_output_dir_forget_me_not_finetuned):
    """
    This test runs the ForgetMeNot algorithm for attention using the safe tensor file
    generated by test_generate_safe_tensors. It checks that the finetuned model directory
    and its subdirectories are created as expected.
    """
    from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
    from mu.algorithms.forget_me_not import ForgetMeNotEvaluator
    from mu.algorithms.forget_me_not.configs import forget_me_not_train_attn_mu, forget_me_not_evaluation_config
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    steps = config['forget_me_not']['steps']
    ti_weights_path = os.path.join(config['forget_me_not']['ti_output_dir'], f"step_inv_{steps}.safetensors")
    if not os.path.exists(ti_weights_path):
        pytest.skip("Safe tensor file not found; skipping run_mu_forget_me_not test.")

    algorithm = ForgetMeNotAlgorithm(
        forget_me_not_train_attn_mu,
        ckpt_path = common_config_unlearn_canvas["diffuser_model_dir"],
        raw_dataset_dir = common_config_unlearn_canvas['unlearn_canvas_data_dir'],
        steps=10, 
        ti_weights_path=ti_weights_path,
        devices=config['forget_me_not']['devices'],
        template=common_config_unlearn_canvas['template'],
        output_dir = config['forget_me_not']['finetuned_output_dir']
    )
    try:
        algorithm.run(train_type="train_attn")
    except Exception as e:
        pytest.fail(f"run_mu_forget_me_not raised an exception: {str(e)}")

    # Expected finetuned model directory
    forget_me_not_output_ckpt_path = config['forget_me_not']['finetuned_output_dir']
    assert os.path.exists(forget_me_not_output_ckpt_path), (
        f"Expected finetuned model directory {forget_me_not_output_ckpt_path} was not created"
    )

    # Check for expected subdirectories and model_index.json
    expected_subdirs = [
        "feature_extractor", "safety_checker", "scheduler", "text_encoder",
        "tokenizer", "unet", "vae"
    ]
    for subdir in expected_subdirs:
        subdir_path = os.path.join(forget_me_not_output_ckpt_path, subdir)
        assert os.path.exists(subdir_path), (
            f"Expected subdirectory {subdir_path} was not created"
        )
    model_index = os.path.join(forget_me_not_output_ckpt_path, "model_index.json")
    assert os.path.exists(model_index), (
        f"Expected model index file {model_index} was not created"
    )

    #run the evaluator
    evaluator = ForgetMeNotEvaluator(
        forget_me_not_evaluation_config,
        ckpt_path = forget_me_not_output_ckpt_path,
    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"Forget me not evalautor raised an exception: {str(e)}")


def test_generate_mask():
    from mu.algorithms.saliency_unlearning.algorithm import MaskingAlgorithm
    from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_generate_mask_mu

    output_dir = config['saliency_unlearning']['mask_dir']
    threshold = config['saliency_unlearning']['threshold']

    masking_algorithm = MaskingAlgorithm(
        saliency_unlearning_generate_mask_mu,
        ckpt_path = common_config_unlearn_canvas['compvis_model_dir'],
        raw_dataset_dir = common_config_unlearn_canvas['unlearn_canvas_data_dir'],
        output_dir = output_dir,
        threshold = threshold,
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template']
    )

    # Run the mask generation step.
    try:
        masking_algorithm.run()
    except Exception as e:
        pytest.fail(f"run_saliency_unlearning mask generation raised an exception: {str(e)}")

    expected_mask_file = os.path.join(output_dir, f"{threshold}.pt")
    
    assert os.path.exists(expected_mask_file), f"Expected mask file {expected_mask_file} was not created"


def test_run_saliency_unlearning(setup_output_dir_saliency_unlearning):
    from mu.algorithms.saliency_unlearning import SaliencyUnlearningEvaluator
    from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
    from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_train_mu, saliency_unlearning_evaluation_config
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    mask_output_dir = config['saliency_unlearning']['mask_dir']
    mask_output_dir = "outputs/saliency_unlearning/masks"
    threshold = config['saliency_unlearning']['threshold']

    algorithm = SaliencyUnlearnAlgorithm(
        saliency_unlearning_train_mu,
        raw_dataset_dir=common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        ckpt_path=common_config_unlearn_canvas["compvis_model_dir"],
        output_dir=config['saliency_unlearning']['output_dir'],
        mask_path = os.path.join(mask_output_dir, f"{threshold}.pt"),
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_saliency_unlearning raised an exception: {str(e)}")

    template_name = common_config_unlearn_canvas["template_name"]
    output_filename = f"saliency_unlearning_{template_name}_model.pth"
    saliency_unlearning_output_ckpt_path = os.path.join(config['saliency_unlearning']['output_dir'], output_filename)
    
    assert os.path.exists(saliency_unlearning_output_ckpt_path), (
        f"Expected output file {saliency_unlearning_output_ckpt_path} was not created"
    )
    assert os.path.isfile(saliency_unlearning_output_ckpt_path), (
        f"{saliency_unlearning_output_ckpt_path} is not a file"
    )
    assert saliency_unlearning_output_ckpt_path.endswith('.pth'), (
        "Output file does not have .pth extension"
    )

    #run the evaluator
    evaluator = SaliencyUnlearningEvaluator(
        saliency_unlearning_evaluation_config,
        ckpt_path = saliency_unlearning_output_ckpt_path,

    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"saliency unlearning evalautor raised an exception: {str(e)}")


def test_run_semipermeable(setup_output_dir_semipermeable):
    from mu.algorithms.semipermeable_membrane.algorithm import SemipermeableMembraneAlgorithm
    from mu.algorithms.semipermeable_membrane import SemipermeableMembraneEvaluator
    from mu.algorithms.semipermeable_membrane.configs import semipermiable_membrane_train_mu, semipermeable_membrane_eval_config
    from evaluation.metrics.accuracy import accuracy_score
    from evaluation.metrics.fid import fid_score

    train_config = config['semipermeable']['train']

    algorithm = SemipermeableMembraneAlgorithm(
        semipermiable_membrane_train_mu,
        output_dir = config['semipermeable']['output_dir'],
        raw_dataset_dir = common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        ckpt_path = common_config_unlearn_canvas["diffuser_model_dir"],
        train = train_config,
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template']
    )
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_semipermeable raised an exception: {str(e)}")

    semiperimable_output_ckpt_path = config['semipermeable']['output_dir']
    pth_files = [f for f in os.listdir(semiperimable_output_ckpt_path) if f.endswith('.safetensors')]
    assert pth_files, f"No .safetensors file found in {semiperimable_output_ckpt_path}"

    ckpt_path = f"{semiperimable_output_ckpt_path}/semipermeable_membrane_{config['common_config_unlearn_canvas_mu']['template_name']}_last.safetensors"
    evaluator = SemipermeableMembraneEvaluator(
    semipermeable_membrane_eval_config,
    spm_path = [ckpt_path],

    )
    try:
        generated_images_path = evaluator.generate_images()
        accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = config['evaluator_config']['classifier_ckpt_path'],
                          forget_theme="Bricks",
                          seed_list = ["188"] )
        fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=config['common_config_unlearn_canvas_mu']['unlearn_canvas_data_dir'],
                 )
    except Exception as e:
        pytest.fail(f"Semipermeable evalautor raised an exception: {str(e)}")



def test_run_selective_anmesia(setup_output_dir_selective_amnesia):
    from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
    from mu.algorithms.selective_amnesia.configs import (
        selective_amnesia_config_quick_canvas,
    )
    selective_amnesia_config_quick_canvas.lightning["trainer"]["max_epochs"] = 1

    algorithm = SelectiveAmnesiaAlgorithm(
        selective_amnesia_config_quick_canvas,
        output_dir = config['selective_amnesia']['output_dir'],
        raw_dataset_dir = common_config_unlearn_canvas["unlearn_canvas_data_dir"],
        ckpt_path = common_config_unlearn_canvas["compvis_model_dir"],
        full_fisher_dict_pkl_path = config['selective_amnesia']['full_fisher_dict_pkl_path'],
        replay_prompt_path = config['selective_amnesia']['replay_prompt_path'],
        use_sample = common_config_unlearn_canvas['use_sample'],
        dataset_type = common_config_unlearn_canvas['dataset_type'],
        template_name=common_config_unlearn_canvas["template_name"],
        template=common_config_unlearn_canvas['template']
    )
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_selective_amnesia raised an exception: {str(e)}")

    output_dir = config['selective_amnesia']['output_dir']
    pth_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    assert pth_files, f"No .pth file found in {output_dir}"


##### ************** TEST WITH i2p DATASET  ******************* ########

def test_run_erase_diff_i2p(setup_output_dir_erase_diff):
    from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
    from mu.algorithms.erase_diff.configs import erase_diff_train_i2p

    algorithm = EraseDiffAlgorithm(
        erase_diff_train_i2p,
        ckpt_path=common_config_i2p["i2p_data_dir"],
        raw_dataset_dir=common_config_i2p["i2p_data_dir"],
        template_name=common_config_i2p["template_name"],
        output_dir=config['erase_diff']['output_dir'],
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_erase_diff raised an exception: {str(e)}")

    output_dir = config['erase_diff']['output_dir']
    template_name = common_config_i2p["template_name"]
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

def test_run_esd_i2p(setup_output_dir_esd):
    from mu.algorithms.esd.algorithm import ESDAlgorithm
    from mu.algorithms.esd.configs import esd_train_i2p

    algorithm = ESDAlgorithm(
        esd_train_i2p,
        ckpt_path=common_config_i2p["i2p_data_dir"],
        raw_dataset_dir=common_config_i2p["i2p_data_dir"],
        template_name=common_config_i2p["template_name"],
        output_dir=config['esd']['output_dir'],
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template=common_config_i2p['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_esd raised an exception: {str(e)}")

    output_dir = config['esd']['output_dir']
    template_name = common_config_i2p["template_name"]
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

def test_run_concept_ablation_i2p(setup_output_dir_concept_ablation):
    from mu.algorithms.concept_ablation.algorithm import ConceptAblationAlgorithm
    from mu.algorithms.concept_ablation.configs import concept_ablation_train_i2p

    concept_ablation_train_i2p.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_i2p,
        ckpt_path=common_config_i2p["i2p_data_dir"],
        prompts=config['concept_ablation']['prompts'],
        output_dir=config["concept_ablation"]["output_dir"],
        raw_dataset_dir=common_config_i2p["unlearn_canvas_data_dir"],
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template']
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

def test_run_scissorhands_i2p(setup_output_dir_scissorhands):
    from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
    from mu.algorithms.scissorhands.configs import scissorhands_train_i2p

    algorithm = ScissorHandsAlgorithm(
        scissorhands_train_i2p,
        ckpt_path=common_config_i2p["i2p_data_dir"],
        raw_dataset_dir=common_config_i2p["unlearn_canvas_data_dir"],
        output_dir=config['scissorhands']['output_dir'],
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_scissorhands raised an exception: {str(e)}")

    output_dir = config['scissorhands']['output_dir']
    template_name = common_config_i2p["template_name"]
    output_filename = f"scissorhands_{template_name}_model.pth"
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

def test_run_unified_concept_editing_i2p(setup_output_dir_unified_concept_editing):
    from mu.algorithms.unified_concept_editing.algorithm import UnifiedConceptEditingAlgorithm
    from mu.algorithms.unified_concept_editing.configs import unified_concept_editing_train_i2p

    # Use the provided checkpoint path for unified concept editing
    algorithm = UnifiedConceptEditingAlgorithm(
        unified_concept_editing_train_i2p,
        ckpt_path = common_config_i2p['diffuser_model_dir'],
        raw_dataset_dir=common_config_i2p["unlearn_canvas_data_dir"],
        output_dir=config['unified_concept_editing']['output_dir'],
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_unified_concept_editing raised an exception: {str(e)}")

    output_dir = config['unified_concept_editing']['output_dir']
    template_name = common_config_i2p["template_name"]
    expected_model_dir = os.path.join(output_dir, f"uce_{template_name}_model")
    
    # Verify that the main output directory was created
    assert os.path.exists(expected_model_dir), (
        f"Expected output directory {expected_model_dir} was not created"
    )
    

    expected_subdirs = [
        "feature_extractor", "safety_checker", "scheduler", "text_encoder",
        "tokenizer", "unet", "vae"
    ]
    for subdir in expected_subdirs:
        subdir_path = os.path.join(expected_model_dir, subdir)
        assert os.path.exists(subdir_path), (
            f"Expected subdirectory {subdir_path} was not created"
        )
    
    model_index = os.path.join(expected_model_dir, "model_index.json")
    assert os.path.exists(model_index), (
        f"Expected model index file {model_index} was not created"
    )

def test_generate_safe_tensors_i2p():
    """
    This test runs the generate_safe_tensors function from the ForgetMeNot algorithm
    and checks that the .safetensors file is created.
    """
    from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
    from mu.algorithms.forget_me_not.configs import forget_me_not_train_ti_i2p

    algorithm = ForgetMeNotAlgorithm(
        forget_me_not_train_ti_i2p,
        ckpt_path = common_config_i2p["diffuser_model_dir"],
        raw_dataset_dir = common_config_i2p['unlearn_canvas_data_dir'],
        steps = config['forget_me_not']['steps'],
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template'],
        output_dir = config['forget_me_not']['ti_output_dir']
    )
    try:
        algorithm.run(train_type="train_ti")
    except Exception as e:
        pytest.fail(f"generate_safe_tensors raised an exception: {str(e)}")

    # Expected safe tensor file
    steps = config['forget_me_not']['steps'] 
    expected_file = os.path.join(config['forget_me_not']['ti_output_dir'], f"step_inv_{steps}.safetensors")
    assert os.path.exists(expected_file), (
        f"Expected safe tensor file {expected_file} was not created"
    )

def test_run_mu_forget_me_not_i2p(setup_output_dir_forget_me_not_finetuned, setup_output_dir_forget_me_not_ti):
    """
    This test runs the ForgetMeNot algorithm for attention using the safe tensor file
    generated by test_generate_safe_tensors. It checks that the finetuned model directory
    and its subdirectories are created as expected.
    """
    from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
    from mu.algorithms.forget_me_not.configs import forget_me_not_train_attn_mu

    steps = config['forget_me_not']['steps']
    ti_weights_path = os.path.join(config['forget_me_not']['ti_output_dir'], f"step_inv_{steps}.safetensors")
    # ti_weights_path = "/home/ubuntu/Projects/Palistha/testing/outputs/forget_me_not/ti_models/step_inv_10.safetensors"
    if not os.path.exists(ti_weights_path):
        pytest.skip("Safe tensor file not found; skipping run_mu_forget_me_not test.")


    algorithm = ForgetMeNotAlgorithm(
        forget_me_not_train_attn_mu,
        ckpt_path = common_config_i2p["diffuser_model_dir"],
        raw_dataset_dir = common_config_i2p['unlearn_canvas_data_dir'],
        steps=steps, 
        ti_weights_path=ti_weights_path,
        devices=config['forget_me_not']['devices'],
        template=common_config_i2p['template'],
        template_name = common_config_i2p['template_name'],
        output_dir = os.path.join(config['forget_me_not']['finetuned_output_dir'], common_config_i2p["template_name"])
    )
    try:
        algorithm.run(train_type="train_attn")
    except Exception as e:
        pytest.fail(f"run_mu_forget_me_not raised an exception: {str(e)}")

    # Expected finetuned model directory
    output_dir = config['forget_me_not']['finetuned_output_dir']
    template_name = common_config_i2p["template_name"]
    expected_model_dir = os.path.join(output_dir, template_name)
    assert os.path.exists(expected_model_dir), (
        f"Expected finetuned model directory {expected_model_dir} was not created"
    )

    # Check for expected subdirectories and model_index.json
    expected_subdirs = [
        "feature_extractor", "safety_checker", "scheduler", "text_encoder",
        "tokenizer", "unet", "vae"
    ]
    for subdir in expected_subdirs:
        subdir_path = os.path.join(expected_model_dir, subdir)
        assert os.path.exists(subdir_path), (
            f"Expected subdirectory {subdir_path} was not created"
        )
    model_index = os.path.join(expected_model_dir, "model_index.json")
    assert os.path.exists(model_index), (
        f"Expected model index file {model_index} was not created"
    )


def test_generate_mask_i2p():
    from mu.algorithms.saliency_unlearning.algorithm import MaskingAlgorithm
    from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_generate_mask_i2p

    output_dir = config['saliency_unlearning']['mask_dir']
    threshold = config['saliency_unlearning']['threshold']

    masking_algorithm = MaskingAlgorithm(
        saliency_unlearning_generate_mask_i2p,
        ckpt_path = common_config_i2p['compvis_model_dir'],
        raw_dataset_dir = common_config_i2p['unlearn_canvas_data_dir'],
        output_dir = output_dir,
        threshold = threshold,
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template']
    )

    # Run the mask generation step.
    try:
        masking_algorithm.run()
    except Exception as e:
        pytest.fail(f"run_saliency_unlearning mask generation raised an exception: {str(e)}")

    expected_mask_file = os.path.join(output_dir, f"{threshold}.pt")
    
    assert os.path.exists(expected_mask_file), f"Expected mask file {expected_mask_file} was not created"


def test_run_saliency_unlearning_i2p(setup_output_dir_saliency_unlearning):
    from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
    from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_train_i2p

    # mask_output_dir = config['saliency_unlearning']['mask_dir']
    mask_output_dir = "test_output/saliency_unlearning/"
    threshold = config['saliency_unlearning']['threshold']

    algorithm = SaliencyUnlearnAlgorithm(
        saliency_unlearning_train_i2p,
        raw_dataset_dir=common_config_i2p["unlearn_canvas_data_dir"],
        ckpt_path=common_config_i2p["i2p_data_dir"],
        output_dir=config['saliency_unlearning']['output_dir'],
        mask_path = os.path.join(mask_output_dir, f"{threshold}.pt"),
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template']
    )
    
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_saliency_unlearning raised an exception: {str(e)}")

    template_name = common_config_i2p["template_name"]
    output_filename = f"saliency_unlearning_{template_name}_model.pth"
    expected_output_file = os.path.join(config['saliency_unlearning']['output_dir'], output_filename)
    
    assert os.path.exists(expected_output_file), (
        f"Expected output file {expected_output_file} was not created"
    )
    assert os.path.isfile(expected_output_file), (
        f"{expected_output_file} is not a file"
    )
    assert expected_output_file.endswith('.pth'), (
        "Output file does not have .pth extension"
    )


def test_run_semipermeable_i2p(setup_output_dir_semipermeable):
    from mu.algorithms.semipermeable_membrane.algorithm import SemipermeableMembraneAlgorithm
    from mu.algorithms.semipermeable_membrane.configs import semipermiable_membrane_train_i2p

    train_config = config['semipermeable']['train']

    algorithm = SemipermeableMembraneAlgorithm(
        semipermiable_membrane_train_i2p,
        output_dir = config['semipermeable']['output_dir'],
        raw_dataset_dir = common_config_i2p["unlearn_canvas_data_dir"],
        ckpt_path = common_config_i2p["diffuser_model_dir"],
        train = train_config,
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template']
    )
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_semipermeable raised an exception: {str(e)}")

    output_dir = config['semipermeable']['output_dir']
    pth_files = [f for f in os.listdir(output_dir) if f.endswith('.safetensors')]
    assert pth_files, f"No .safetensors file found in {output_dir}"


def test_run_selective_amnesia_i2p(setup_output_dir_selective_amnesia):
    from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
    from mu.algorithms.selective_amnesia.configs import (
        selective_amnesia_config_i2p,
    )
    selective_amnesia_config_i2p.lightning["trainer"]["max_epochs"] = 1

    algorithm = SelectiveAmnesiaAlgorithm(
        selective_amnesia_config_i2p,
        output_dir = config['selective_amnesia']['output_dir'],
        raw_dataset_dir = common_config_i2p["unlearn_canvas_data_dir"],
        ckpt_path = common_config_i2p["i2p_data_dir"],
        full_fisher_dict_pkl_path = config['selective_amnesia']['full_fisher_dict_pkl_path'],
        replay_prompt_path = config['selective_amnesia']['replay_prompt_path'],
        use_sample = common_config_i2p['use_sample'],
        dataset_type = common_config_i2p['dataset_type'],
        template_name=common_config_i2p["template_name"],
        template=common_config_i2p['template']
    )
    try:
        algorithm.run()
    except Exception as e:
        pytest.fail(f"run_selective_amnesia raised an exception: {str(e)}")

    output_dir = config['selective_amnesia']['output_dir']
    pth_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    assert pth_files, f"No .pth file found in {output_dir}"

if __name__ == "__main__":
    pytest.main([__file__])

