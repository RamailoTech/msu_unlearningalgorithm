
"""
class MachineUnlearningDataset:
			    Dataset for Machine Unlearning. This dataset will provide access to styles and object classes
			    for model training, including handling image loading and transformations.
						    
			    Args:
			        image_dir (str): Path to the directory containing images.
			        styles (list[str]): List of available styles (e.g., Van Gogh, Picasso).
			        object_classes (list[str]): List of object classes (e.g., Dogs, Cats).
			        transform (callable, optional): Optional transform to be applied on a sample image.
			        shuffle (bool, optional): Whether to shuffle the dataset on initialization.

			    methods: 
			    	transform 
			    	_load_image
			    	_generate_prompt
			    	__getitem__ : Return a single sample from the dataset.
"""