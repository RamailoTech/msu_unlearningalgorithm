# mu_attack/core/mu_attack_evaluator.py

from mu_attack.core import BaseEvaluator
from evaluation.evaluators.asr import ASREvaluator
from evaluation.evaluators.clip_score import ClipScoreEvaluator
from evaluation.evaluators.mu_attack_fid import FIDEvaluator

class MuAttackEvaluator(BaseEvaluator):
    def _parse_config(self, **kwargs):
        """
        Optionally override or extend the configuration using kwargs.
        Here, we simply update attributes on the config.
        """
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def run_evaluation(self):
        """
        Runs all the evaluators (ASR, CLIP Score, and FID) in sequence
        and aggregates their results into self.results.
        """
        results = {}

        # Run ASR Evaluator
        asr_evaluator = ASREvaluator(
            config=self.config,
            root=self.config.asr.root,
            root_no_attack=self.config.asr.root_no_attack,
            output_path=self.config.output_path
        )
        print("Running ASR Evaluator...")
        asr_evaluator.run()
        results['ASR'] = asr_evaluator.results

        # Run CLIP Score Evaluator
        clip_evaluator = ClipScoreEvaluator(
            config=self.config,
            image_path=self.config.clip.image_path,
            log_path=self.config.clip.log_path,
            output_path=self.config.output_path,
            devices=self.config.clip.devices
        )
        print("Running CLIP Score Evaluator...")
        clip_evaluator.run()
        results['CLIP'] = clip_evaluator.result

        # Run FID Evaluator
        fid_evaluator = FIDEvaluator(
            config=self.config,
            ref_batch_path=self.config.fid.ref_batch_path,
            sample_batch_path=self.config.fid.sample_batch_path,
            output_path=self.config.output_path
        )
        print("Running FID Evaluator...")
        fid_evaluator.run()
        results['FID'] = fid_evaluator.result

        self.results = results
        return results

    def run(self):
        """
        Executes the full evaluation process.
        """
        return self.run_evaluation()
