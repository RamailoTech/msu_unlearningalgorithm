# mu_attack/configs/nudity/seed_search_esd_nudity_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class SeedSearchESDNudityClassifierCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="seed_search",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        attack_idx=1,
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/seed_search_esd_nudity_scissorhands", 
              "name": "Seed Search Nudity"}
    )

seed_search_esd_nudity_classifier_compvis_config = SeedSearchESDNudityClassifierCompvis()
