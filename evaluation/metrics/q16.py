# evaluation/metrics/q16.py

import torch
from PIL import Image

from mu_attack.tasks.utils.metrics.harm_eval import load_prompts, SimClassifier, compute_embeddings, ClipWrapper

def q16_binary_classifier(device, prompt_path='mu_attack/tasks/utils/metrics/q16/prompts.p'):
    trained_prompts = load_prompts(prompt_path, device=device)
    clip_model = ClipWrapper(device)
    classifier = SimClassifier(trained_prompts, device)
    classifier.to(device)
    return clip_model, classifier

def harm_eval(clip_model, classifier, img_path, device):
    image = Image.open(img_path)
    x = compute_embeddings(clip_model, image, device)
    y = classifier(x)
    label = torch.argmax(y, dim=0) # 0 for benign, 1 for malicious
    return label.item(), torch.softmax(y, dim=-1).squeeze()
    