from typing import Dict, Iterable, List, Optional, Union, Tuple
from PIL import Image
import numpy as np 
import torch 

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token): 
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(
    image: Image, 
    size: Tuple[int, int], 
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None
) -> Image: 
    height, width = size 
    resize_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    
    return resize_image

def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray: 
    rescaled_image = image * scale 
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image 

def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[Iterable[float]]) -> np.ndarray: 
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    normalized_image = (image - mean) / std 
    return normalized_image 

def process_images(
    images: List[Image.Image], 
    size: Tuple[int, int] = None,
    resample: Image.Resampling = None, 
    rescale_factor: float = None, 
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None
) -> List[np.ndarray]: 
    height, width = size[0], size[1]
    images = [resize(image=image, size=[height, width], resample=resample) for image in images]
    images = [np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    images = [image.transpose(2, 0, 1) for image in images]
    return images 

class PaliGemmaProcessor: 
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size 
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.tokenizer = tokenizer
        
    def __call__(self, 
                text: List[str], 
                images: List[Image.Image], 
                padding: str = "longest",
                truncation: bool = True,
                add_special_tokens: bool = False 
                ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."
        
        pixel_values = process_images(
            images, 
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )   
        
        
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_to_prompt(
                prefix_prompt=prompt, 
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]
        
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data