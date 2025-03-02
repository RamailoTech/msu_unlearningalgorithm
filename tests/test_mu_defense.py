import os
import shutil
import pytest
import yaml

# Load configuration from YAML file
with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)
common_config_unlearn_canvas = config["common_config_unlearn_canvas_mu"]

@pytest.fixture
def setup_output_dir_adv_unlearn():
    output_dir = config.get("adv_unlearn", {}).get("output_dir", "results/adv_unlearn")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield output_dir
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

def test_adv_unlearn_run_compvis(setup_output_dir_adv_unlearn,run_erase_diff):
    from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
    from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config, mu_defense_evaluation_config
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu
    from mu_defense.algorithms.adv_unlearn import MUDefenseEvaluator
    from mu_defense.algorithms.adv_unlearn.configs import example_image_generator_config
    from mu_defense.algorithms.adv_unlearn import ImageGenerator
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu
        
    output_dir = config['erase_diff']['output_dir']
    template_name = common_config_unlearn_canvas["template_name"]
    output_filename = f"erase_diff_{template_name}_model.pth"
    compvis_model_checkpoint = os.path.join(output_dir, output_filename)

    try:
        mu_defense = AdvUnlearnAlgorithm(
            config=adv_unlearn_config,
            compvis_ckpt_path = compvis_model_checkpoint,  # path to the finetuned model
            attack_step = config['mu_defense']['attack_step'],
            backend = "compvis",
            attack_method = config['mu_defense']['attack_method'],
            train_method = config['mu_defense']['train_method'],  # training method; see docs for available options
            warmup_iter = config['mu_defense']['warmup_iter'],
            iterations = config['mu_defense']['iterations'],
            model_config_path=erase_diff_train_mu.model_config_path  # use the same model config path for the used model
        )
        mu_defense.run()
    except Exception as e:
        pytest.fail(f"AdvUnlearnAlgorithm raised an exception: {str(e)}")

    output_dir = os.path.join(adv_unlearn_config.output_dir,"models")
    files = os.listdir(output_dir)
    pt_files = [file for file in files if file.endswith('.pt')]
    assert pt_files, f"No .pt files were found in the output directory {output_dir}. Files present: {files}"

    output_ckpt_path = f"{adv_unlearn_config.output_dir}/models/Diffusers-UNet-noxattn-epoch_0.pt"
    generate_image = ImageGenerator(
        config = example_image_generator_config,
        target_ckpt =output_ckpt_path,
        model_config_path = erase_diff_train_mu.model_config_path,
        save_path = "test_output/adv_unlearn/models",
        prompts_path = "data/prompts/sample_prompt.csv",
        num_samples = 1,
        folder_suffix = "imagenette",
        devices = "0",

    )
    try:
        generate_image.generate_images()
    except Exception as e:
        pytest.fail(f"Image generation for mu_defense evalaution raised an exception: {str(e)}")

    evaluator = MUDefenseEvaluator(
        config = mu_defense_evaluation_config,
        gen_imgs_path = config['mu_defense']['evaluation']['gen_imgs_path'],
        coco_imgs_path = config['mu_defense']['evaluation']['coco_imgs_path'],
        output_path = "test_output/adv_unlearn/evaluation/",
    )

    try:
        evaluator.run()
    except Exception as e:
        pytest.fail(f"mu_defense evalaution raised an exception: {str(e)}")


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

def test_adv_unlearn_run_diffusers(setup_output_dir_adv_unlearn, forget_me_not_model_output):
    import os
    from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
    from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config

    forget_me_not_model_output = config['forget_me_not']['finetuned_output_dir']
    try:
        mu_defense = AdvUnlearnAlgorithm(
            config = adv_unlearn_config,
            diffusers_model_name_or_path = forget_me_not_model_output,  # path to the finetuned model
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
        "text_encoder",
        "tokenizer",
        "unet",
        "vae"
    ]
    expected_file = "model_index.json"
    output_dir = adv_unlearn_config.output_dir

    for folder in expected_folders:
        folder_path = os.path.join(output_dir, folder)
        assert os.path.isdir(folder_path), f"Expected folder '{folder}' not found in {output_dir}."

    file_path = os.path.join(output_dir, expected_file)
    assert os.path.isfile(file_path), f"Expected file '{expected_file}' not found in {output_dir}."



def test_adv_unlearn_run_diffusers_without_text_encoder(setup_output_dir_adv_unlearn, forget_me_not_model_output):
    from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
    from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config

    forget_me_not_model_output = config['forget_me_not']['finetuned_output_dir']
    try:
        mu_defense = AdvUnlearnAlgorithm(
            config = adv_unlearn_config,
            # Use the finetuned model directory generated from ForgetMeNot as the input model
            diffusers_model_name_or_path = forget_me_not_model_output,
            attack_step = config['mu_defense']['attack_step'],
            backend = "diffusers",
            attack_method = config['mu_defense']['attack_method'],
            train_method = config['mu_defense']['train_method'],
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
        "text_encoder",
        "tokenizer",
        "unet",
        "vae"
    ]
    expected_file = "model_index.json"
    output_dir = adv_unlearn_config.output_dir

    for folder in expected_folders:
        folder_path = os.path.join(output_dir, folder)
        assert os.path.isdir(folder_path), f"Expected folder '{folder}' not found in {output_dir}."

    file_path = os.path.join(output_dir, expected_file)
    assert os.path.isfile(file_path), f"Expected file '{expected_file}' not found in {output_dir}."

