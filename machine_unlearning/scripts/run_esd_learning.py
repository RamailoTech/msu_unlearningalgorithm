from machine_learning.algorithms.esd.esd_model import ESDModel
from machine_learning.algorithms.esd.esd_trainer import ESDTrainer
from machine_learning.algorithms.esd.esd_sampler import ESDSampler
from machine_learning.algorithms.esd.esd_algorithm import ESDAlgorithm

# Load the model, sampler, and trainer
model = ESDModel(config_path='path_to_config.yaml', ckpt_path='path_to_checkpoint.ckpt', device='cuda')
sampler = ESDSampler(model)
trainer = ESDTrainer(model, learning_rate=1e-5, iterations=1000)

# Build the ESD algorithm
esd_algorithm = ESDAlgorithm.build(model=model, trainer=trainer, sampler=sampler)

# Perform unlearning
esd_algorithm.unlearn(prompt="Van Gogh")

# Save the model after unlearning
esd_algorithm.save_model(save_path="path_to_save_model.pth")

# Load the saved model
esd_algorithm.load_model(model_path="path_to_saved_model.pth")
