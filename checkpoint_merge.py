"""
Stable Diffusion Checkpoint Merger CLI 
==================================
This module provides functionalities to merge Stable Diffusion models and perform inferences.

Credits:
-------
- Modified from: https://github.com/painebenjamin/app.enfugue.ai/blob/main/src/python/enfugue/diffusion/util/model_util.py
- Inspired by: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/dev/modules/extras.py
"""


import os
import re
import gc
import argparse
import random
import torch
import safetensors.torch
from PIL import Image
from typing import Optional, Union, Literal, Dict, cast
torch._dynamo.config.suppress_errors = True
from infer import Inference
import yaml 
import numpy as np

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
class ModelMerger:
    """
    Allows merging various Stable Diffusion models of various sizes.
    Inspired by https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/extras.py
    """
    CHECKPOINT_DICT_REPLACEMENTS = {
        "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
        "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
        "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
    }
    CHECKPOINT_DICT_SKIP = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]
    discard_weights: Optional[re.Pattern]

    def __init__(
            self,
            models_config,
            interpolation: Optional[Literal["add-difference", "weighted-sum"]] = None,
            multiplier: Union[int, float] = 1.0,
            half: bool = True,
            discard_weights: Optional[Union[str, re.Pattern]] = None,
    ):
        self.models_config = models_config
        self.interpolation = interpolation
        self.multiplier = multiplier
        self.half = half

        self.discard_weights = re.compile(discard_weights) if isinstance(discard_weights, str) else discard_weights

    @staticmethod
    def as_half(tensor: torch.Tensor) -> torch.Tensor:
        """Halves a tensor if necessary"""
        return tensor.half() if tensor.dtype == torch.float else tensor

    @staticmethod
    def get_difference(theta0: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
        """Returns the difference between two tensors."""
        return theta0 - theta1
    
    @staticmethod
    def weighted_sum(theta0: torch.Tensor, theta1: torch.Tensor, alpha: Union[int, float]) -> torch.Tensor:
        """Returns the weighted sum of two tensors."""
        return ((1 - alpha) * theta0) + (alpha * theta1)

    @staticmethod
    def add_weighted_difference(theta0: torch.Tensor, theta1: torch.Tensor, alpha: Union[int, float]) -> torch.Tensor:
        """Adds a weighted difference back to the original tensor."""
        return theta0 + (alpha * theta1)
    
    @staticmethod
    def dare_merge_tensors(tensor1, tensor2, mask_p):
     # Calculate the delta of the weights
        delta = tensor2 - tensor1
        # Generate the mask m^t from Bernoulli distribution
        m = torch.from_numpy(np.random.binomial(1, mask_p, delta.shape)).to(tensor1.dtype)
        # Apply the mask to the delta to get δ̃^t
        delta_tilde = m * delta
        # Scale the masked delta by the dropout rate to get δ̂^t
        delta_hat = delta_tilde / (1 - mask_p)
        return delta_hat

    @staticmethod
    def get_state_dict_from_checkpoint(checkpoint: Dict) -> Dict:
        """Extracts the state dictionary from the checkpoint."""
        state_dict = checkpoint.pop("state_dict", checkpoint)
        state_dict.pop("state_dict", None)  # Remove any sub-embedded state dicts

        transformed_dict = {
            ModelMerger.transform_checkpoint_key(key): value for key, value in state_dict.items()
        }
        return transformed_dict

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict:
        """Loads the checkpoint's state dictionary."""
        _, ext = os.path.splitext(checkpoint_path)
        print(f"Loading {checkpoint_path}...")
        checkpoint = safetensors.torch.load_file(
            checkpoint_path, device="cpu") if ext.lower() == ".safetensors" else torch.load(checkpoint_path, map_location="cpu")
        return ModelMerger.get_state_dict_from_checkpoint(checkpoint)

    @staticmethod
    def is_ignored_key(key: str) -> bool:
        """Checks if a key should be ignored during merge."""
        return "model" not in key or key in ModelMerger.CHECKPOINT_DICT_SKIP

    @staticmethod
    def transform_checkpoint_key(text: str) -> str:
        """Transforms a checkpoint key if needed."""
        for key, value in ModelMerger.CHECKPOINT_DICT_REPLACEMENTS.items():
            if key.startswith(text):
                return value + text[len(key):]
        return text

    def norm_weight(self,model_config):
        allweight=0
        for model_info in model_config:
            allweight+=model_info['weight']
        for i in range(len(model_config)):
            model_config[i]['weight']= model_config[i]['weight']/allweight
        return model_config
        
    def merge_models(self, output_path: str) -> None:
        """Runs the configured merger."""
        prev_state_dict=None
        interpolate = self.add_weighted_difference 
        merged_keys = set()
        self.models_config=self.norm_weight(self.models_config)

        for model_info in self.models_config:
            modelpath=model_info['path']
            weight_scale=model_info['weight']
            cur_state_dict=self.load_checkpoint(modelpath)
            if prev_state_dict is not None:            # weight sum
                theta_0= prev_state_dict
                theta_1= cur_state_dict
                for key in prev_state_dict.keys():
                    # if "cond_stage_model" in key:# 文本编码器在key中
                    #     continue
                    if "first_stage_model" in key :      # Vae 跳过，当有些模型没有vae时，当有些模型没有vae时,需要使用
                        continue
                    if key not in theta_1 or self.is_ignored_key(key):
                        continue
                    a, b = theta_0[key], theta_1[key]
                    # Check for different model types based on channel count
                    if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                        self._handle_different_model_types(a, b, key, theta_0, interpolate,weight_scale)
                    else:
                        theta_0[key] = interpolate(a, b, weight_scale)
                    if self.half:
                        theta_0[key] = self.as_half(theta_0[key])
                del theta_1 
                prev_state_dict= theta_0
            else:         # 第一个模型，只进行缩放
                for key in cur_state_dict.keys():
                    if self.is_ignored_key(key):
                        continue
                    if "first_stage_model" in key :      # Vae 跳过，当有些模型没有vae时,需要进行跳过
                        continue
                    cur_state_dict[key]=weight_scale*cur_state_dict[key]
                    merged_keys.add(key)  # Add the key to the set
                prev_state_dict=cur_state_dict

        if self.discard_weights:
            theta_0 = {key: value for key, value in theta_0.items() if not re.search(self.discard_weights, key)}

        _, extension = os.path.splitext(output_path)
        if extension.lower() == ".safetensors":
            safetensors.torch.save_file(theta_0, output_path)
        else:
            torch.save(theta_0, output_path)
        
        # Save merged keys to a local file
        keys_output_path = "merged_keys.txt"  # File to save the keys
        with open(keys_output_path, "w") as f:
            for key in sorted(merged_keys):
                f.write(key + "\n")
        print(f"Saving Merged Model:")
        print(f"  - Path: {output_path}")
        print(f"  - File Format: {'SafeTensors' if output_path.endswith('.safetensors') else 'PyTorch'}\n")

    def _handle_different_model_types(self, a, b, key, theta_0, interpolate,weight):
        """Handles the case when merging different types of models."""
        if a.shape[1] == 4 and (b.shape[1] == 9 or b.shape[1] == 8):
            raise RuntimeError(
                "When merging different types of models, the primary model must be the specialized one."
            )

        if a.shape[1] == 8 and b.shape[1] == 4:
            theta_0[key][:, 0:4, :, :] = interpolate(a[:, 0:4, :, :], b, weight)
        else:
            assert a.shape[1] == 9 and b.shape[1] == 4, f"Unexpected dimensions for merged layer {key}: A={a.shape}, B={b.shape}"
            theta_0[key][:, 0:4, :, :] = interpolate(a[:, 0:4, :, :], b, weight)

def parse_paths(input_string):
    return [path.strip() for path in input_string.split(',')]# 将输入字符串按逗号分隔，去除多余空格
 
def parse_arguments():
    """Parses command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Merge and optionally validate Stable Diffusion models.")
    
    parser.add_argument("yaml_path", default='config.yaml',type=str, help="输入各个模型检查点的路径，用逗号分隔。")
    parser.add_argument("--interpolation", choices=["add-difference", "weighted-sum"], help="Interpolation method to use.")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Multiplier for the interpolation.")
    parser.add_argument("--half", action="store_true", help="Halve the tensor if necessary.")
    parser.add_argument("--discard_weights", help="Pattern of weights to discard.")
    parser.add_argument("--prompt", help="Positive prompts for the validation.")
    parser.add_argument("--image_output", help="Path to save the generated image.")
    parser.add_argument("--sampler", default="Euler a", help="Scheduler sampler for the validation. Defaults to 'Euler a'.")
    parser.add_argument("--sdxl", action="store_true", help="Use StableDiffusionXLPipeline instead of StableDiffusionPipeline.")
    parser.add_argument("--disable_torch_compile", action="store_true", help="Disable torch.compile for the unet model.")
    parser.add_argument("--validate", action="store_true", help="validate model.")

    return parser.parse_args()

def checkpoint_merger(
    yaml_config: list,
    interpolation: Optional[Literal["add-difference", "weighted-sum"]] = None,
    multiplier: float = 1.0,
    half: bool = False,
    discard_weights: Optional[str] = None,
    validate: bool = False,              # validate in dataset
    prompt: Optional[str] = None,        # test by prompt in dataset
    image_output: Optional[str] = None,
    sampler: str = "Euler a",
    sdxl: bool = False,
    disable_torch_compile: bool = False,
):
    models_config=yaml_config['models']
    # Notify users about loading the checkpoint
    print("Loading checkpoint...")
    print(f"Checkpoint: {models_config}\n")
    
    merger = ModelMerger(
        models_config=models_config,
        interpolation=interpolation,
        multiplier=multiplier,
        half=half,
        discard_weights=discard_weights
    )

    print("Merging Models:")
    print(f"  - Models:   {models_config}")
    print(f"Multiplier: {multiplier}")
    print(f"Tensor Halving: {'Enabled' if half else 'Disabled'}")
    print(f"Discard Weights: {discard_weights}\n")
    
    output_dir=yaml_config['param']['output_dir']
    if os.path.isdir(output_dir):
        output_path=os.path.join(output_dir,'PicLumen_H1.5_Realistic.safetensors')
    else:
        output_path=output_dir
    merger.merge_models(output_path)
    free_memory()
    
    if prompt:
        infer = Inference(output_path, sdxl=sdxl, disable_torch_compile=disable_torch_compile)
        images = infer.validate(prompt, sdxl=sdxl)
        free_memory()
    
        if image_output:
            basename = os.path.splitext(os.path.basename(image_output))[0]
            dirpath = os.path.dirname(image_output)
            
            if isinstance(images, list):
                for idx, img in enumerate(images):
                    img.save(os.path.join(dirpath, f'{basename}_{idx}.png'))
            else:
                images.save(image_output)
    
        return images

def main():
    args = parse_arguments()
    with open(args.yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)

    checkpoint_merger(
        yaml_config=yaml_config['config'],
        interpolation=args.interpolation,
        multiplier=args.multiplier,
        half=args.half,
        discard_weights=args.discard_weights,
        validate=args.validate,
        prompt=args.prompt,
        image_output=args.image_output,
        sampler=args.sampler,
        sdxl=args.sdxl,
        disable_torch_compile=args.disable_torch_compile,
    )

if __name__ == "__main__":
    main()
    free_memory()