### Run Train <br>

Each algorithm has their own script to run the algorithm, Some also have different process all together. Follow usage section in readme for the algorithm you want to run with the help of the github repository. You will need to run the code snippet provided in usage section with necessary configuration passed. 


**Example usage for erase_diff algorithm (CompVis model)**

**Note: Currently we only have support for unlearn canvas dataset. I2p and generic dataset support needs to be added.**

The default configuration for training is provided by erase_diff_train_mu. You can run the training with the default settings as follows:

**Using the Default Configuration**

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_mu

algorithm = EraseDiffAlgorithm(
    erase_diff_train_mu
)
algorithm.run()
```

<br> <br>

**Overriding the Default Configuration**

If you need to override the existing configuration settings, you can specify your custom parameters (such as ckpt_path and raw_dataset_dir) directly when initializing the algorithm. For example:

**Machine unlearning using unlearn canvas dataset:**


```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_mu

algorithm = EraseDiffAlgorithm(
    erase_diff_train_mu,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt", #replace it with your ckpt path
    raw_dataset_dir="data/quick-canvas-dataset/sample",
    use_sample = True, #uses sample dataset
    template_name = "Abstractionism",
    dataset_type = "unlearncanvas",
    devices = "0"
)
algorithm.run()
```

<span style="color: red;"><br>Note: When fine-tuning the model, if you want to use a sample dataset, set use_sample=True (default).Otherwise, set use_sample=False to use the full dataset.<br></span>

**Note**
You can choose from a set of predefined `template_name` options to erase specific concept when working with the `unlearncanvas` dataset to perform unlearning. For instance, in the i2p context, the available choices include:

```
"Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
 "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
 "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
 "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
 "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
 "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
 "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
 "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"
```

**Machine unlearning with i2p dataset**

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_i2p

algorithm = EraseDiffAlgorithm(
    erase_diff_train_i2p,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt", #replace it with your ckpt path
    raw_dataset_dir="data/i2p-dataset/sample",
    num_samples = 1,
    dataset_type = "i2p",
    template = "i2p",
    template_name = "self-harm", #concept to erase
    use_sample = True, #uses sample dataset
    devices = "0"
    
)
algorithm.run()
```

**Note**
You can choose from a set of predefined `template_name` options to erase specific concept when working with the `i2p` dataset to perform unlearning. For instance, in the i2p context, the available choices include:

 `'shocking', 'harassment', 'hate', 'self-harm', 'sexual', 'illegal activity', 'violence'`


### **Add your own unlearning algorithms:**

For detailed instructions on adding your own algorithm, please see the [Contribution](#contribution-section) section.
