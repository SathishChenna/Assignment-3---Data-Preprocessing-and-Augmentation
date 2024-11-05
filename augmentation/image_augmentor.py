import cv2
import numpy as np
from PIL import Image
import random

class ImageAugmentor:
    def __init__(self):
        self.operations = {
            "apply_all": self.apply_all_augmentations,
            "rotate": self.rotate_image,
            "flip": self.flip_image,
            "brightness": self.adjust_brightness,
            "contrast": self.adjust_contrast,
            "noise": self.add_noise
        }
    
    def get_available_operations(self):
        return list(op for op in self.operations.keys() if op != "apply_all")
    
    def apply_operation(self, operation: str, image: np.ndarray) -> np.ndarray:
        if operation in self.operations:
            return self.operations[operation](image)
        return image

    def apply_all_augmentations(self, image: np.ndarray) -> np.ndarray:
        augmented_image = image.copy()
        # Apply all operations except 'apply_all'
        for op_name, op_func in self.operations.items():
            if op_name != "apply_all":
                augmented_image = op_func(augmented_image)
        return augmented_image

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        angle = random.uniform(-30, 30)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return rotated_image
    
    def flip_image(self, image: np.ndarray) -> np.ndarray:
        flip_directions = [0, 1, -1]
        flip_type = random.choice(flip_directions)
        flipped_image = image.copy()
        return cv2.flip(flipped_image, flip_type)
    
    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        brightness = random.uniform(0.5, 1.5)
        return cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    def adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        contrast = random.uniform(0.5, 1.5)
        return cv2.convertScaleAbs(image, alpha=contrast, beta=128*(1-contrast))
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return np.clip(noisy_image, 0, 255)