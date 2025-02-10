
import random
import pandas as pd

import torch
import torch.nn.functional as F

from mu.helpers import load_model_from_config, sample_model
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler


class PromptDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.unseen_indices = list(self.data.index)  # 保存所有未见过的索引

    def get_random_prompts(self, num_prompts=1):
        # Ensure that the number of prompts requested is not greater than the number of unseen prompts
        num_prompts = min(num_prompts, len(self.unseen_indices))

        # Randomly select num_prompts indices from the list of unseen indices
        selected_indices = random.sample(self.unseen_indices, num_prompts)
        
        # Remove the selected indices from the list of unseen indices
        for index in selected_indices:
            self.unseen_indices.remove(index)

        # return the prompts corresponding to the selected indices
        return self.data.loc[selected_indices, 'prompt'].tolist()

    def has_unseen_prompts(self):
        # check if there are any unseen prompts
        return len(self.unseen_indices) > 0
    
    def reset(self):
        self.unseen_indices = list(self.data.index)
        
    def check_unseen_prompt_count(self):
        return len(self.unseen_indices)

def retain_prompt(dataset_retain):
    # Prompt Dataset to be retained

    if dataset_retain == 'imagenet243':
        retain_dataset = PromptDataset('data/prompts/train/imagenet243_retain.csv')
    elif dataset_retain == 'imagenet243_no_filter':
        retain_dataset = PromptDataset('data/prompts/train/imagenet243_no_filter_retain.csv')
    elif dataset_retain == 'coco_object':
        retain_dataset = PromptDataset('data/prompts/train/coco_object_retain.csv')
    elif dataset_retain == 'coco_object_no_filter':
        retain_dataset = PromptDataset('data/prompts/train/coco_object_no_filter_retain.csv')
    else:
        raise ValueError('Invalid dataset for retaining prompts')
    
    return retain_dataset

def id2embedding(tokenizer, all_embeddings, input_ids, device):
    input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(tokenizer.get_vocab())).float()
    input_one_hot = torch.unsqueeze(input_one_hot,0).to(device)
    input_embeds = input_one_hot @ all_embeddings
    return input_embeds

def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler


def get_train_loss_retain( retain_batch, retain_train, retain_loss_w, model, model_orig, text_encoder, sampler, emb_0, emb_p, retain_emb_p,  emb_n, retain_emb_n, start_guidance, negative_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type, adv_embd=None):
    """_summary_

    Args:
        model: ESD model
        model_orig: frozen DDPM model
        sampler: DDIMSampler for DDPM model
        
        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)
        
        start_guidance: unconditional guidance for ESD model
        negative_guidance: negative guidance for ESD model
        
        devices: list of devices for ESD and DDPM models 
        ddim_steps: number of steps for DDIMSampler
        ddim_eta: eta for DDIMSampler
        image_size: image size for DDIMSampler
        
        criteria: loss function for ESD model
        
        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding

    Returns:
        loss: training loss for ESD model
    """
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, n_samples=batch, till_T=t, verbose=False)
    
    
    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
    if retain_train == 'reg':
        retain_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])
    
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, 1, int(t_enc)) # emb_p seems to work better instead of emb_0
        # get conditional and unconditional scores from frozen model at time step t and image z
        e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
        e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
        
        if retain_train == 'reg':
            retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch, int(t_enc)) # emb_p seems to work better instead of emb_0
            # retain_e_0 = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_0.to(devices[0]))
            retain_e_p = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_p.to(devices[0]))
    
    if adv_embd is None:
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))
    else:
        if attack_embd_type == 'condition_embd':
            # Train with adversarial conditional embedding
            e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), adv_embd.to(devices[0]))
        elif attack_embd_type == 'word_embd':
            # Train with adversarial word embedding
            print('====== Training with adversarial word embedding =====')
            adv_emb_n = text_encoder(input_ids = adv_input_ids.to(devices[0]), inputs_embeds=adv_embd.to(devices[0]))[0]
            e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), adv_emb_n.to(devices[0]))
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')
    
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
    # loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) 
    
    # return loss

    if retain_train == 'reg':
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        print('====== Training with retain batch =====')
        unlearn_loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) 
        
        retain_e_n = model.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_n.to(devices[0]))
        
        # retain_e_0.requires_grad = False
        retain_e_p.requires_grad = False
        retain_loss = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))
        
        loss = unlearn_loss + retain_loss_w * retain_loss
        return loss
        
    else:
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        unlearn_loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) 
        return unlearn_loss
    

def param_choices(model, train_method, component='all', final_layer_norm=False):
    # choose parameters to train based on train_method
    parameters = []
    
    # Text Encoder FUll Weight Tuning
    if train_method == 'text_encoder_full':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Final Layer Norm
            if name.startswith('final_layer_norm'):
                if component == 'all' or final_layer_norm==True:
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            # Transformer layers 
            elif name.startswith('encoder'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            # Embedding layers
            else:
                pass
           
    # Text Encoder Layer 0 Tuning
    elif train_method == 'text_encoder_layer0':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer012':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer01234':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer012345':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123456':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01234567':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012345678':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123456789':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012345678910':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01234567891011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer0_11':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    
    elif train_method == 'text_encoder_layer01_1011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012_91011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass

    # UNet Model Tuning
    else:
        for name, param in model.model.diffusion_model.named_parameters():
            # train all layers except x-attns and time_embed layers
            if train_method == 'noxattn':
                if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                    pass
                else:
                    print(name)
                    parameters.append(param)
                    
            # train only self attention layers
            if train_method == 'selfattn':
                if 'attn1' in name:
                    print(name)
                    parameters.append(param)
                    
            # train only x attention layers
            if train_method == 'xattn':
                if 'attn2' in name:
                    print(name)
                    parameters.append(param)
                    
            # train all layers
            if train_method == 'full':
                print(name)
                parameters.append(param)
                
            # train all layers except time embed layers
            if train_method == 'notime':
                if not (name.startswith('out.') or 'time_embed' in name):
                    print(name)
                    parameters.append(param)
            if train_method == 'xlayer':
                if 'attn2' in name:
                    if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                        print(name)
                        parameters.append(param)
            if train_method == 'selflayer':
                if 'attn1' in name:
                    if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                        print(name)
                        parameters.append(param)
    
    return parameters