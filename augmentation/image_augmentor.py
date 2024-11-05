import cv2
import numpy as np
from PIL import Image
import random

class ImageAugmentor:
    def __init__(self):
        self.operations = {
            "rotate": self.rotate_image,
            "flip": self.flip_image,
            "brightness": self.adjust_brightness,
            "contrast": self.adjust_contrast,
            "noise": self.add_noise
        }
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, image: np.ndarray) -> np.ndarray:
        if operation in self.operations:
            return self.operations[operation](image)
        return image

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        # Get the image dimensions
        height, width = image.shape[:2]
        
        # Calculate the center of the image
        center = (width // 2, height // 2)
        
        # Generate a random angle between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to avoid cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust the rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform the rotation and return the image
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # White background
        )
        
        return rotated_image
    
    def flip_image(self, image: np.ndarray) -> np.ndarray:
        # Randomly choose flip direction:
        # 0: vertical flip
        # 1: horizontal flip
        # -1: both horizontal and vertical flip
        flip_directions = [0, 1, -1]
        flip_type = random.choice(flip_directions)
        
        # Create a copy of the image to avoid modifying the original
        flipped_image = image.copy()
        
        # Apply the flip
        flipped_image = cv2.flip(flipped_image, flip_type)
        
        return flipped_image
    
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