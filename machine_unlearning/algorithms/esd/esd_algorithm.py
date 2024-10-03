class ESDAlgorithm(UnlearningAlgorithm):
    """
    ESD-specific implementation of the UnlearningAlgorithm.
    """

    def __init__(self, model, trainer, sampler):
        self.model = model
        self.trainer = trainer
        self.sampler = sampler

    @staticmethod
    def build(model, trainer, sampler, **kwargs):
        return ESDAlgorithm(model, trainer, sampler)

    def unlearn(self, prompt, **kwargs):
        print(f"Starting ESD unlearning for: {prompt}")
        self.trainer.train(prompt, self.sampler, **kwargs)
        print(f"Unlearning completed for: {prompt}")

    def save_model(self, save_path):
        torch.save(self.model.model.state_dict(), save_path)
        print(f"Model saved to {save_path}.")

    def load_model(self, model_path):
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}.")
