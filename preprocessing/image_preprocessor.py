import cv2
import numpy as np
from PIL import Image
import io

class ImagePreprocessor:
    def __init__(self):
        self.operations = {
            "grayscale": self.to_grayscale,
            "resize": self.resize_image,
            "normalize": self.normalize,
            "blur": self.apply_blur,
            "sharpen": self.sharpen
        }
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, image: np.ndarray) -> np.ndarray:
        if operation in self.operations:
            return self.operations[operation](image)
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        # Resize to a fixed size (e.g., 224x224)
        return cv2.resize(image, (224, 224))
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def array_to_bytes(image: np.ndarray) -> bytes:
        # Convert numpy array to bytes for sending to frontend
        success, encoded_image = cv2.imencode('.png', image)
        return encoded_image.tobytes() 