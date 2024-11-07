import cv2
import numpy as np
from PIL import Image
import io

class ImagePreprocessor:
    def __init__(self):
        self.operations = {
            "apply_all": self.apply_all_preprocessing,
            "grayscale": self.to_grayscale,
            "resize": self.resize_image,
            "normalize": self.normalize,
            "blur": self.apply_blur,
            "sharpen": self.sharpen,
            "equalize_histogram": self.equalize_histogram,
            "remove_noise": self.remove_noise,
            "detect_edges": self.detect_edges
        }
    
    def get_available_operations(self):
        return list(op for op in self.operations.keys() if op != "apply_all")
    
    def apply_operation(self, operation: str, image: np.ndarray) -> np.ndarray:
        if operation in self.operations:
            return self.operations[operation](image)
        return image

    def apply_all_preprocessing(self, image: np.ndarray) -> np.ndarray:
        processed_image = image.copy()
        for op_name, op_func in self.operations.items():
            if op_name != "apply_all":
                processed_image = op_func(processed_image)
        return processed_image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (224, 224))
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.equalizeHist(image)
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        try:
            # Check if image is colored or grayscale
            if len(image.shape) == 3:
                # For colored images
                return cv2.fastNlMeansDenoisingColored(
                    image,
                    None,
                    h=10,  # Filter strength (5-12 is a good range)
                    hColor=10,  # Same as h for colored image
                    templateWindowSize=7,  # Should be odd (3,5,7 are good values)
                    searchWindowSize=21  # Should be odd (21 is a good value)
                )
            else:
                # For grayscale images
                return cv2.fastNlMeansDenoising(
                    image,
                    None,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
        except Exception as e:
            print(f"Error in noise removal: {str(e)}")
            # If denoising fails, return original image
            return image
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, 100, 200)

    @staticmethod
    def array_to_bytes(image: np.ndarray) -> bytes:
        success, encoded_image = cv2.imencode('.png', image)
        return encoded_image.tobytes()