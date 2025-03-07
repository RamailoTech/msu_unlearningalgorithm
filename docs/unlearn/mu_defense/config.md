#### Sample config for Advattack (mu_defense)

```python
#mu_defense/algorithms/adv_unlearn/configs/adv_unlearn_config.py

import os
from pathlib import Path
from mu_defense.core.base_config import BaseConfig


class AdvUnlearnConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Inference & Model Paths
        self.model_config_path = "configs/stable-diffusion/v1-inference.yaml" #for compvis
        self.compvis_ckpt_path = "models/sd-v1-4-full-ema.ckpt"
        self.encoder_model_name_or_path = "CompVis/stable-diffusion-v1-4"
        self.cache_path = ".cache"

        self.diffusers_model_name_or_path = ""
        self.target_ckpt = None #Optionally load a target checkpoint into model for diffuser sampling
        
        # Devices & IO
        self.devices = "0,0"  # You can later parse this string into a list if needed.
        self.seperator = None
        self.output_dir = "outputs/adv_unlearn"
        
        # Image & Diffusion Sampling
        self.image_size = 512
        self.ddim_steps = 50
        self.start_guidance = 3.0
        self.negative_guidance = 1.0

        # Training Setup
        self.prompt = "nudity"
        self.dataset_retain = "coco_object"  # Choices: 'coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'
        self.retain_batch = 5
        self.retain_train = "iter"  # Options: 'iter' or 'reg'
        self.retain_step = 1
        self.retain_loss_w = 1.0
        self.ddim_eta = 0

        self.train_method = "text_encoder_full"   #choices: text_encoder_full', 'text_encoder_layer0', 'text_encoder_layer01', 'text_encoder_layer012', 'text_encoder_layer0123', 'text_encoder_layer01234', 'text_encoder_layer012345', 'text_encoder_layer0123456', 'text_encoder_layer01234567', 'text_encoder_layer012345678', 'text_encoder_layer0123456789', 'text_encoder_layer012345678910', 'text_encoder_layer01234567891011', 'text_encoder_layer0_11','text_encoder_layer01_1011', 'text_encoder_layer012_91011', 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer
        self.norm_layer = False  # This is a flag; use True if you wish to update the norm layer.
        self.attack_method = "pgd"  # Choices: 'pgd', 'multi_pgd', 'fast_at', 'free_at'
        self.component = "all"     # Choices: 'all', 'ffn', 'attn'
        self.iterations = 10
        self.save_interval = 200
        self.lr = 1e-5

        # Adversarial Attack Hyperparameters
        self.adv_prompt_num = 1
        self.attack_embd_type = "word_embd"  # Choices: 'word_embd', 'condition_embd'
        self.attack_type = "prefix_k"         # Choices: 'replace_k', 'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'
        self.attack_init = "latest"           # Choices: 'random', 'latest'
        self.attack_step = 30
        self.attack_init_embd = None
        self.adv_prompt_update_step = 1
        self.attack_lr = 1e-3
        self.warmup_iter = 200

        #backend
        self.backend = "compvis"

        # Override default values with any provided keyword arguments.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if self.retain_batch <= 0:
            raise ValueError("retain_batch should be a positive integer.")
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if self.image_size <= 0:
            raise ValueError("Image size should be a positive integer.")
        if self.iterations <= 0:
            raise ValueError("Iterations must be a positive integer.")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

adv_unlearn_config = AdvUnlearnConfig()
```

### Description of fields in config file

Below is a detailed description of the configuration fields available in the `adv_unlearn_config.py` file. The descriptions match those provided in the help section of the command-line arguments.

1. **Inference & Model Paths**

   * **model_config_path**  
     *Description*: Config path for stable diffusion model. Use for compvis model only. 
     *Type*: `str`  
     *Example*: `configs/stable-diffusion/v1-inference.yaml`

   * **compvis_ckpt_path**  
     *Description*: Checkpoint path for stable diffusion v1-4.  
     *Type*: `str`  
     *Example*: `models/sd-v1-4-full-ema.ckpt`

   * **encoder_model_name_or_path**  
     *Description*: Model name or path for the encoder.
     *Type*: `str`  
     *Example*: `CompVis/stable-diffusion-v1-4`

   * **cache_path**  
     *Description*: Directory used for caching model files.  
     *Type*: `str`  
     *Example*: `.cache`

   * **diffusers_model_name_or_path**  
     *Description*: Model name or path for the diffusers (if used).  
     *Type*: `str`  
     *Example*: `outputs/forget_me_not/finetuned_models/Abstractionism`

   * **target_ckpt**  
     *Description*: Optionally load a target checkpoint into the model for diffuser sampling.  
     *Type*: Typically `str` or `None`  
     *Example*: `path to target checkpoint path`

2. **Devices & IO**

   * **devices**  
     *Description*: CUDA devices to train on.  
     *Type*: `str`  
     *Example*: `0,0`

   * **seperator**  
     *Description*: Separator used if you want to train a bunch of words separately.  
     *Type*: `str` or `None`  
     *Example*: `None`

   * **output_dir**  
     *Description*: Directory where output files (e.g., checkpoints, logs) are saved.  
     *Type*: `str`  
     *Example*: `outputs/adv_unlearn`

3. **Image & Diffusion Sampling**

   * **image_size**  
     *Description*: Image size used to train.  
     *Type*: `int`  
     *Example*: `512`

   * **ddim_steps**  
     *Description*: Number of DDIM steps for inference during training.  
     *Type*: `int`  
     *Example*: `50`

   * **start_guidance**  
     *Description*: Guidance of start image used to train.  
     *Type*: `float`  
     *Example*: `3.0`

   * **negative_guidance**  
     *Description*: Guidance of negative training used to train.  
     *Type*: `float`  
     *Example*: `1.0`

   * **ddim_eta**  
     *Description*: DDIM eta parameter for sampling.  
     *Type*: `int` or `float`  
     *Example*: `0`

4. **Training Setup**

   * **prompt**  
     *Description*: Prompt corresponding to the concept to erase.  
     *Type*: `str`  
     *Example*: `nudity`

   * **dataset_retain**  
     *Description*: Prompts corresponding to non-target concepts to retain.  
     *Type*: `str`  
     *Choices*: `coco_object`, `coco_object_no_filter`, `imagenet243`, `imagenet243_no_filter`  
     *Example*: `coco_object`

   * **retain_batch**  
     *Description*: Batch size of retaining prompts during training.  
     *Type*: `int`  
     *Example*: `5`

   * **retain_train**  
     *Description*: Retaining training mode; choose between iterative (`iter`) or regularization (`reg`).  
     *Type*: `str`  
     *Choices*: `iter`, `reg`  
     *Example*: `iter`

   * **retain_step**  
     *Description*: Number of steps for retaining prompts.  
     *Type*: `int`  
     *Example*: `1`

   * **retain_loss_w**  
     *Description*: Retaining loss weight.  
     *Type*: `float`  
     *Example*: `1.0`

   * **train_method**  
     *Description*: Method of training.  
     *Type*: `str`  
     *Choices*:  
       `text_encoder_full`, `text_encoder_layer0`, `text_encoder_layer01`, `text_encoder_layer012`, `text_encoder_layer0123`, `text_encoder_layer01234`, `text_encoder_layer012345`, `text_encoder_layer0123456`, `text_encoder_layer01234567`, `text_encoder_layer012345678`, `text_encoder_layer0123456789`, `text_encoder_layer012345678910`, `text_encoder_layer01234567891011`, `text_encoder_layer0_11`, `text_encoder_layer01_1011`, `text_encoder_layer012_91011`, `noxattn`, `selfattn`, `xattn`, `full`, `notime`, `xlayer`, `selflayer`  
     *Example*: `text_encoder_full`

   * **norm_layer**  
     *Description*: Flag indicating whether to update the norm layer during training.  
     *Type*: `bool`  
     *Example*: `False`

   * **attack_method**  
     *Description*: Method for adversarial attack training.  
     *Type*: `str`  
     *Choices*: `pgd`, `multi_pgd`, `fast_at`, `free_at`  
     *Example*: `pgd`

   * **component**  
     *Description*: Component to apply the attack on.  
     *Type*: `str`  
     *Choices*: `all`, `ffn`, `attn`  
     *Example*: `all`

   * **iterations**  
     *Description*: Total number of training iterations.  
     *Type*: `int`  
     *Example*: `10`  
     *(Note: The help argument may default to a higher value, e.g., 1000, but the config file sets it to 10.)*

   * **save_interval**  
     *Description*: Interval (in iterations) at which checkpoints are saved.  
     *Type*: `int`  
     *Example*: `200`

   * **lr**  
     *Description*: Learning rate used during training.  
     *Type*: `float`  
     *Example*: `1e-5`

5. **Adversarial Attack Hyperparameters**

   * **adv_prompt_num**  
     *Description*: Number of prompt tokens for adversarial soft prompt learning.  
     *Type*: `int`  
     *Example*: `1`

   * **attack_embd_type**  
     *Description*: The adversarial embedding type; options are word embedding or condition embedding.  
     *Type*: `str`  
     *Choices*: `word_embd`, `condition_embd`  
     *Example*: `word_embd`

   * **attack_type**  
     *Description*: The type of adversarial attack applied to the prompt.  
     *Type*: `str`  
     *Choices*: `replace_k`, `add`, `prefix_k`, `suffix_k`, `mid_k`, `insert_k`, `per_k_words`  
     *Example*: `prefix_k`

   * **attack_init**  
     *Description*: Strategy for initializing the adversarial attack; either randomly or using the latest parameters.  
     *Type*: `str`  
     *Choices*: `random`, `latest`  
     *Example*: `latest`

   * **attack_step**  
     *Description*: Number of steps for the adversarial attack.  
     *Type*: `int`  
     *Example*: `30`

   * **attack_init_embd**  
     *Description*: Initial embedding for the attack (optional).  
     *Type*: Depends on implementation; default is `None`  
     *Example*: `None`

   * **adv_prompt_update_step**  
     *Description*: Frequency (in iterations) at which the adversarial prompt is updated.  
     *Type*: `int`  
     *Example*: `1`

   * **attack_lr**  
     *Description*: Learning rate for adversarial attack training.  
     *Type*: `float`  
     *Example*: `1e-3`

   * **warmup_iter**  
     *Description*: Number of warmup iterations before starting the adversarial attack.  
     *Type*: `int`  
     *Example*: `200`

6. **Backend**

   * **backend**  
     *Description*: Backend framework to be used (e.g., CompVis).  
     *Type*: `str`  
     *Example*: `compvis`
     *Choices*: `compvis` or `diffusers`


## Directory Structure

- `algorithm.py`: Implementation of the AdvUnlearnAlgorithm class.
- `configs/`: Contains configuration files for AdvUnlearn for compvis and diffusers.
- `model.py`: Implementation of the AdvUnlearnModel class for compvis and diffusers.
- `trainer.py`: Trainer for adversarial unlearning for compvis and diffusers.
- `utils.py`: Utility functions used in the project.
- `dataset_handler.py`: handles prompt cleaning and retaining dataset creation for adversarial unlearning.
- `compvis_trainer.py`: Trainer for adversarial unlearning for compvis.
- `diffusers_trainer.py`: Trainer for adversarial unlearning for diffusers.




