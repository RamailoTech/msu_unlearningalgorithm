# unified_concept_algorithm.py

from base_algorithm import BaseAlgorithm

class UnifiedConceptEditingAlgorithm(BaseAlgorithm):
    def __init__(self, config):
        self.config = config

        # Initialize DataHandler
        self.data_handler = UnifiedConceptDataHandler()

        # Initialize Model
        self.model = UnifiedConceptModel()
        self.model.load_model(
            config_path=config.get('config_path', ''),
            ckpt_path=config['ckpt_path'],
            device=config.get('device', 'cuda')
        )

        # Initialize Trainer
        self.trainer = UnifiedConceptTrainer(
            model=self.model,
            config=config
        )

    def run(self):
        old_texts = self.config['old_texts']
        new_texts = self.config['new_texts']
        retain_texts = self.config.get('retain_texts', [])
        approach = self.config.get('approach', 'erasing')

        self.trainer.train(old_texts, new_texts, retain_texts, approach=approach)

        output_path = self.config.get('output_path', 'edited_model')
        self.trainer.save_checkpoint(output_path)
