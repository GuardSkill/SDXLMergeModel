import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline,AutoPipelineForText2Image

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    # _, extension = os.path.splitext(checkpoint_file)
    # if extension.lower() == ".safetensors":
    #     device = map_location

    #     if not shared.opts.disable_mmap_load_safetensors:
    #         pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
    #     else:
    #         pl_sd = safetensors.torch.load(open(checkpoint_file, 'rb').read())
    #         pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    # else:
    #     pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    # if print_global_state and "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    pipeline = StableDiffusionPipeline.from_single_file( checkpoint_file,
    torch_dtype=torch.float16, variant="fp16") #AutoPipeline 
    unet = pipeline.unet
    print(unet)
    # AutoPipeline 
    pipeline = pipeline.to("cuda")
    pl_sd=pipeline
    sd = get_state_dict_from_checkpoint(pl_sd)

    return sd    

def get_state_dict_from_checkpoint(pl_sd):
    # pl_sd = pl_sd.pop("state_dict", pl_sd)
    # pl_sd.pop("state_dict", None)

    # is_sd2_turbo = 'conditioner.embedders.0.model.ln_final.weight' in pl_sd and pl_sd['conditioner.embedders.0.model.ln_final.weight'].size()[0] == 1024

    # sd = {}
    # for k, v in pl_sd.items():
    #     if is_sd2_turbo:
    #         new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd2_turbo)
    #     else:
    #         new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd1)

    #     if new_key is not None:
    #         sd[new_key] = v

    # pl_sd.clear()
    # pl_sd.update(sd)
    return pl_sd

def merge_models(models_list, device='cuda'):
    loaded_models = []
    weights = []
    
    
    # Load and extract state dictionaries from each model
    for model_path in models_list:
        state_dict = read_state_dict(model_path)
        loaded_models.append(state_dict)
        
        # Calculate number of parameters for weight normalization
        num_params = sum(p.numel() for p in state_dict.values())
        weights.append(num_params)

    # Normalize weights
    total_params = sum(weights)
    weights = [np.float32(num_param / total_params) for num_param in weights]

    # Merge models
    merged_model=None
    # merged_model = models.DiffuserStack(
    #     [models.load_model_from_state_dict(model, device=device) for model in loaded_models],
    #     weights=weights,
    #     normalize_post=True
    # )

    return merged_model

def run_modelmerger(models_list, output_path):
    # Merge models
    merged_model = merge_models(models_list)

    # Save the merged model
    torch.save(merged_model.state_dict(), output_path)

# Example usage:
models_list = [
    # "E:/Models/SD1.5/realism/Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
               'D:\Projects\stable-diffusion-webui\models\Stable-diffusion/majicmixRealistic_v7.safetensors',
                'D:\Projects\stable-diffusion-webui\models\Stable-diffusion\epicphotogasm_ultimateFidelity.safetensors', 
                'D:\Projects\stable-diffusion-webui\models\Stable-diffusion\artErosAerosATribute_aerosNovae.safetensors']
output_path = 'merged_model.pth'
run_modelmerger(models_list, output_path)
