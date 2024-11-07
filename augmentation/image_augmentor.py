import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import io
import random

class ImageAugmentor:
    def __init__(self):
        self.operations = {
            "rotate": self.rotate_image,
            "flip": self.flip_image,
            "brightness": self.adjust_brightness,
            "contrast": self.adjust_contrast,
            "noise": self.add_noise,
            "random_crop": self.random_crop,
            "random_erase": self.random_erase
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, image_bytes: bytes) -> bytes:
        if operation in self.operations:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            # Convert PIL Image to Tensor
            image_tensor = F.to_tensor(image).to(self.device)
            # Apply operation
            augmented_tensor = self.operations[operation](image_tensor)
            # Convert back to PIL Image
            augmented_image = F.to_pil_image(augmented_tensor.cpu())
            # Convert to bytes
            buffer = io.BytesIO()
            augmented_image.save(buffer, format='PNG')
            return buffer.getvalue()
        return image_bytes

    def rotate_image(self, image: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-30, 30)
        return F.rotate(image, angle, fill=1.0)
    
    def flip_image(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            image = F.hflip(image)
        if random.random() > 0.5:
            image = F.vflip(image)
        return image
    
    def adjust_brightness(self, image: torch.Tensor) -> torch.Tensor:
        brightness_factor = random.uniform(0.5, 1.5)
        return F.adjust_brightness(image, brightness_factor)
    
    def adjust_contrast(self, image: torch.Tensor) -> torch.Tensor:
        contrast_factor = random.uniform(0.5, 1.5)
        return F.adjust_contrast(image, contrast_factor)
    
    def add_noise(self, image: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(image) * 0.1
        return torch.clamp(image + noise, 0, 1)
    
    def random_crop(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.shape
        new_h = int(h * 0.8)
        new_w = int(w * 0.8)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        return F.crop(image, top, left, new_h, new_w)
    
    def random_erase(self, image: torch.Tensor) -> torch.Tensor:
        transform = transforms.RandomErasing(
            p=1.0,
            scale=(0.02, 0.4),
            ratio=(0.3, 3.3),
            value=random.random()
        )
        return transform(image)