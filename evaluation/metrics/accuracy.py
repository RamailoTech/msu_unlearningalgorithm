# evaluation/metrics/accuracy.py

import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.constants.const import theme_available, class_available

from mu.datasets.constants import i2p_categories
from evaluation.helpers.utils import preprocess_image, load_categories, load_model



def accuracy_score(gen_image_dir, 
                    dataset_type,
                    classifier_ckpt_path,
                    forget_theme,
                    device = "0",
                    classification_model = "vit_large_patch16_224",
                    task = "class",
                    reference_dir=None,  #if using generic dataset then provide prompt path to extract categories
                    seed_list=[188, 288, 588, 688, 888]):
    """
    Calculate accuracy (and related metrics) for generated images for different dataset types.
    
    Args:
        config (dict): Configuration dictionary containing paths and settings.
        dataset_type (str): The type of dataset. Options: "unlearncanvas", "i2p", "generic".
        model (torch.nn.Module): The classification model.
        preprocess_image (func): A function to preprocess PIL images.
        device (str): The device to use (e.g. "cuda" or "cpu").
        reference_dir (str): path to generic dataset for prompt_csv path
    
    Returns:
        tuple: (results (dict), eval_output_path (str))
    """
    #TODO assert datatype as it works for unlearn only for now.

    device = [
            f"cuda:{int(d.strip())}" for d in device.split(",")
        ][0]
    model = load_model(classifier_ckpt_path, device,classification_model,task)
    input_dir = gen_image_dir 

    # For the original datasets, modify input_dir based on theme if applicable.
    if task in ["style", "class"]:
        theme = forget_theme
        if theme is not None:
            input_dir = os.path.join(input_dir, theme)


    results = {}
    if dataset_type == "unlearncanvas":
        if task == "style":
            results = {
                "test_theme": forget_theme,
                "input_dir": input_dir,
                "loss": {th: 0.0 for th in theme_available},
                "acc": {th: 0.0 for th in theme_available},
                "pred_loss": {th: 0.0 for th in theme_available},
                "misclassified": {th: {oth: 0 for oth in theme_available} for th in theme_available}
            }
            for idx, test_theme in tqdm(enumerate(theme_available), total=len(theme_available)):
                theme_label = idx
                for seed in seed_list:
                    for object_class in class_available:
                        img_file = f"{test_theme}_{object_class}_seed{seed}.jpg"
                        img_path = os.path.join(input_dir, img_file)
                        if not os.path.exists(img_path):
                            continue
                        image = Image.open(img_path)
                        tensor_img = preprocess_image(device, image)
                        with torch.no_grad():
                            res = model(tensor_img)
                            label = torch.tensor([theme_label]).to(device)
                            loss = F.cross_entropy(res, label)
                            res_softmax = F.softmax(res, dim=1)
                            pred_loss_val = res_softmax[0][theme_label]
                            pred_label = torch.argmax(res)
                            pred_success = (pred_label == theme_label).sum()
                        results["loss"][test_theme] += loss.item()
                        results["pred_loss"][test_theme] += (pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val)
                        results["acc"][test_theme] += (pred_success * 1.0 / (len(class_available) * len(seed_list)))
                        misclassified_as = theme_available[pred_label.item()]
                        results["misclassified"][test_theme][misclassified_as] += 1

        elif task == "class":
            results = {
                "loss": {cls_: 0.0 for cls_ in class_available},
                "acc": {cls_: 0.0 for cls_ in class_available},
                "pred_loss": {cls_: 0.0 for cls_ in class_available},
                "misclassified": {cls_: {other_cls: 0 for other_cls in class_available} for cls_ in class_available}
            }
            for test_theme in tqdm(theme_available, total=len(theme_available)):
                for seed in seed_list:
                    for idx, object_class in enumerate(class_available):
                        label_val = idx
                        img_file = f"{test_theme}_{object_class}_seed_{seed}.jpg"
                        img_path = os.path.join(input_dir, img_file)
                        if not os.path.exists(img_path):
                            continue
                        image = Image.open(img_path)
                        tensor_img = preprocess_image(device, image)
                        label = torch.tensor([label_val]).to(device)
                        with torch.no_grad():
                            res = model(tensor_img)
                            loss = F.cross_entropy(res, label)
                            res_softmax = F.softmax(res, dim=1)
                            pred_loss_val = res_softmax[0][label_val]
                            pred_label = torch.argmax(res)
                            pred_success = (pred_label == label_val).sum()
                        results["loss"][object_class] += loss.item()
                        results["pred_loss"][object_class] += (pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val)
                        results["acc"][object_class] += (pred_success * 1.0 / (len(class_available) * len(seed_list)))
                        misclassified_as = class_available[pred_label.item()]
                        results["misclassified"][object_class][misclassified_as] += 1

    elif dataset_type == "i2p":
        categories = i2p_categories
        results = {
            "loss": {cat: 0.0 for cat in categories},
            "acc": {cat: 0.0 for cat in categories},
            "pred_loss": {cat: 0.0 for cat in categories},
            "misclassified": {cat: {other_cat: 0 for other_cat in categories} for cat in categories},
            "input_dir": input_dir
        }
        for category in tqdm(categories, total=len(categories)):
            for seed in seed_list:
                img_file = f"{category}_seed_{seed}.jpg"
                img_path = os.path.join(input_dir, img_file)
                if not os.path.exists(img_path):
                    continue
                image = Image.open(img_path)
                tensor_img = preprocess_image(device, image)
                label_idx = categories.index(category)
                label = torch.tensor([label_idx]).to(device)
                with torch.no_grad():
                    res = model(tensor_img)
                    loss = F.cross_entropy(res, label)
                    res_softmax = F.softmax(res, dim=1)
                    pred_loss_val = res_softmax[0][label_idx]
                    pred_label = torch.argmax(res)
                    pred_success = (pred_label == label_idx).sum()
                results["loss"][category] += loss.item()
                results["pred_loss"][category] += (pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val)
                results["acc"][category] += (pred_success * 1.0 / len(seed_list))
                misclassified_as = categories[pred_label.item()]
                results["misclassified"][category][misclassified_as] += 1

    elif dataset_type == "generic":
        categories = load_categories(reference_dir)
        results = {
            "loss": {cat: 0.0 for cat in categories},
            "acc": {cat: 0.0 for cat in categories},
            "pred_loss": {cat: 0.0 for cat in categories},
            "misclassified": {cat: {other_cat: 0 for other_cat in categories} for cat in categories},
            "input_dir": input_dir
        }
        for category in tqdm(categories, total=len(categories)):
            for seed in seed_list:
                img_file = f"{category}_seed_{seed}.jpg"
                img_path = os.path.join(input_dir, img_file)
                if not os.path.exists(img_path):
                    continue
                image = Image.open(img_path)
                tensor_img = preprocess_image(device, image)
                label_idx = categories.index(category)
                label = torch.tensor([label_idx]).to(device)
                with torch.no_grad():
                    res = model(tensor_img)
                    loss = F.cross_entropy(res, label)
                    res_softmax = F.softmax(res, dim=1)
                    pred_loss_val = res_softmax[0][label_idx]
                    pred_label = torch.argmax(res)
                    pred_success = (pred_label == label_idx).sum()
                results["loss"][category] += loss.item()
                results["pred_loss"][category] += (pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val)
                results["acc"][category] += (pred_success * 1.0 / len(seed_list))
                misclassified_as = categories[pred_label.item()]
                results["misclassified"][category][misclassified_as] += 1

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return results

