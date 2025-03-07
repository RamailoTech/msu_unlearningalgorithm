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
    templete="class",
    dataset_type = "unlearncanvas",
    devices = "0"
)
algorithm.run()
```

<span style="color: red;"><br>Note: If you want to use a sample dataset for testing purposes, set use_sample=True (default).Otherwise, set use_sample=False to use the full dataset. <br></span>

**Note**
You can choose from a set of predefined `template_name` options to erase specific concept when working with the `unlearncanvas` dataset to perform unlearning. For instance, in the unlearncanvas dataset context, the available choices include:

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

For example if you want to train using `Abstractionism` then you can pass `template_name = "Abstractionism"`.

Also you can choose type of `template` to use during training. The available choices are `style` and `class`.


### **Add your own unlearning algorithms:**

For detailed instructions on adding your own algorithm, please see the [Contribution](/evaluation/contributing/) section.
