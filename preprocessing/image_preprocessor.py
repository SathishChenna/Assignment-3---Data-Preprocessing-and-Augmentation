import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import io
import base64

class ImagePreprocessor:
    def __init__(self):
        self.operations = {
            "grayscale": self.to_grayscale,
            "resize": self.resize_image,
            "normalize": self.normalize,
            "blur": self.apply_blur,
            "sharpen": self.sharpen,
            "equalize_histogram": self.equalize_histogram,
            "remove_noise": self.remove_noise,
            "detect_edges": self.detect_edges
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
            processed_tensor = self.operations[operation](image_tensor)
            # Convert back to PIL Image
            processed_image = F.to_pil_image(processed_tensor.cpu())
            # Convert to bytes
            buffer = io.BytesIO()
            processed_image.save(buffer, format='PNG')
            return buffer.getvalue()
        return image_bytes

    def to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_grayscale(image, num_output_channels=3)
    
    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.resize(image, [224, 224], antialias=True)
    
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def apply_blur(self, image: torch.Tensor) -> torch.Tensor:
        return transforms.GaussianBlur(kernel_size=5, sigma=2.0)(image)
    
    def sharpen(self, image: torch.Tensor) -> torch.Tensor:
        sharpness_factor = 2.0
        return F.adjust_sharpness(image, sharpness_factor)

    def equalize_histogram(self, image: torch.Tensor) -> torch.Tensor:
        # Convert to PIL for equalization
        image_pil = F.to_pil_image(image)
        equalized = transforms.functional.equalize(image_pil)
        return F.to_tensor(equalized)
    
    def remove_noise(self, image: torch.Tensor) -> torch.Tensor:
        # Using bilateral filter for noise reduction
        denoise = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
        return denoise(image)
    
    def detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale first
        gray = F.rgb_to_grayscale(image, num_output_channels=1)
        # Apply Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).float()
        
        # Add batch and channel dimensions
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        gray = gray.unsqueeze(0)  # Add batch dimension
        edges_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        
        # Combine edges
        edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2))
        edges = edges.squeeze(0)  # Remove batch dimension
        
        # Normalize to [0, 1]
        edges = edges / edges.max()
        
        # Convert to 3 channels
        return edges.repeat(3, 1, 1)