def run_adv_unlearn():
    from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
    from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        compvis_ckpt_path="/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/erase_diff/erase_diff_Abstractionism_model.pth",
        diffusers_model_name_or_path="/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/forget_me_not/finetuned_models/Abstractionism",
        attack_step=2,
        backend="diffusers",
        attack_method="fast_at",
        warmup_iter=1,
        iterations=2,
        model_config_path=erase_diff_train_mu.model_config_path,
    )
    mu_defense.run()


def run_concept_ablation():
    from mu.algorithms.concept_ablation.algorithm import ConceptAblationAlgorithm
    from mu.algorithms.concept_ablation.configs import concept_ablation_train_mu

    concept_ablation_train_mu.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_mu,
        config_path="/home/ubuntu/Projects/balaram/msu_unlearningalgorithm/mu/algorithms/concept_ablation/configs/train_config.yaml",
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
        prompts="/home/ubuntu/Projects/balaram/msu_unlearningalgorithm/mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt",
        output_dir="/opt/dlami/nvme/outputs",
    )
    algorithm.run()


def run_unified_concept_editing():
    from mu.algorithms.unified_concept_editing.algorithm import (
        UnifiedConceptEditingAlgorithm,
    )
    from mu.algorithms.unified_concept_editing.configs import (
        unified_concept_editing_train_mu,
    )

    algorithm = UnifiedConceptEditingAlgorithm(
        unified_concept_editing_train_mu,
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50/",
        raw_dataset_dir="/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample",
        output_dir="/opt/dlami/nvme/outputs",
    )
    algorithm.run()


def run_scissorhands():
    from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
    from mu.algorithms.scissorhands.configs import scissorhands_train_mu

    algorithm = ScissorHandsAlgorithm(
        scissorhands_train_mu,
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
        raw_dataset_dir="/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample",
        output_dir="/opt/dlami/nvme/outputs",
    )
    algorithm.run()


def run_selective_amnesia():
    from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
    from mu.algorithms.selective_amnesia.configs import (
        selective_amnesia_config_quick_canvas,
    )

    algorithm = SelectiveAmnesiaAlgorithm(
        selective_amnesia_config_quick_canvas,
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
        raw_dataset_dir="/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample",
    )
    algorithm.run()


def run_erase_diff():
    from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    algorithm = EraseDiffAlgorithm(
        erase_diff_train_mu,
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
        raw_dataset_dir="/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample",
        train_method="noxattn",
    )
    algorithm.run()


def run_esd():
    from mu.algorithms.esd.algorithm import ESDAlgorithm
    from mu.algorithms.esd.configs import esd_train_mu

    algorithm = ESDAlgorithm(
        esd_train_mu,
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
        raw_dataset_dir="/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample",
        train_method="noxattn",
    )
    algorithm.run()


def run_forget_me_not_ti():
    from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
    from mu.algorithms.forget_me_not.configs import (
        forget_me_not_train_attn_mu,
    )

    algorithm = ForgetMeNotAlgorithm(
        forget_me_not_train_attn_mu,
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50",
        raw_dataset_dir=(
            "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
        ),
        steps=10,
        ti_weights_path="outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors",
    )
    algorithm.run(train_type="train_ti")


def run_forget_me_not_attn():
    from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
    from mu.algorithms.forget_me_not.configs import (
        forget_me_not_train_attn_mu,
    )

    algorithm = ForgetMeNotAlgorithm(
        forget_me_not_train_attn_mu,
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50",
        raw_dataset_dir=(
            "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
        ),
        steps=10,
        ti_weights_path="outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors",
    )
    algorithm.run(train_type="train_attn")


def run_saliency_unlearning():
    from mu.algorithms.saliency_unlearning.algorithm import (
        SaliencyUnlearnAlgorithm,
    )
    from mu.algorithms.saliency_unlearning.configs import (
        saliency_unlearning_train_mu,
    )

    algorithm = SaliencyUnlearnAlgorithm(
        saliency_unlearning_train_mu,
        raw_dataset_dir=(
            "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
        ),
        ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
        output_dir="/opt/dlami/nvme/outputs",
    )
    algorithm.run()


def run_semipermeable():
    from mu.algorithms.semipermeable_membrane.algorithm import (
        SemipermeableMembraneAlgorithm,
    )
    from mu.algorithms.semipermeable_membrane.configs import (
        semipermiable_membrane_train_mu,
        SemipermeableMembraneConfig,
    )

    algorithm = SemipermeableMembraneAlgorithm(
        semipermiable_membrane_train_mu,
        output_dir="/opt/dlami/nvme/outputs",
        train={"iterations": 2},
    )
    algorithm.run()


def run_attack_for_nudity():
    from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_compvis_config
    from mu_attack.execs.attack import MUAttack

    overridable_params = {
        "task.compvis_ckpt_path": "/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
        "task.compvis_config_path": "mu/algorithms/scissorhands/configs/model_config.yaml",
        "task.dataset_path": "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/outputs/dataset/i2p_nude",
        "logger.json.root": "results/hard_prompt_esd_nudity_P4D_scissorhands",
    }

    MUAttack(config=hard_prompt_esd_nudity_P4D_compvis_config, **overridable_params)


def run_mu_defense_compvis():
    from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
    from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        compvis_ckpt_path="/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/erase_diff/erase_diff_Abstractionism_model.pth",
        # diffusers_model_name_or_path = "/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/forget_me_not/finetuned_models/Abstractionism",
        attack_step=2,
        # backend = "diffusers",
        backend="compvis",
        attack_method="fast_at",
        train_method="noxattn",
        warmup_iter=1,
        iterations=1,
        model_config_path=erase_diff_train_mu.model_config_path,
    )
    mu_defense.run()


def run_mu_defense_diffuser():
    from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
    from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
    from mu.algorithms.erase_diff.configs import erase_diff_train_mu

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        diffusers_model_name_or_path="/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/forget_me_not/finetuned_models/Abstractionism",
        attack_step=2,
        backend="diffuser",
        attack_method="fast_at",
        train_method="noxattn",
        warmup_iter=1,
        iterations=1,
    )
    mu_defense.run()


if __name__ == "__main__":
    # run_erase_diff()
    # run_esd()
    # run_concept_ablation()
    # run_forget_me_not_ti()
    # run_forget_me_not_attn()
    #     python -m mu.algorithms.saliency_unlearning.scripts.generate_mask \
    # --config_path mu/algorithms/saliency_unlearning/configs/mask_config.yaml
    # run_saliency_unlearning()
    # run_scissorhands()
    # run_attack_for_nudity()
    # run_mu_defense_compvis()
    # run_mu_defense_diffuser()  # TODO

    run_selective_amnesia()  # blocker
    # run_semipermeable()
    # run_unified_concept_editing()
