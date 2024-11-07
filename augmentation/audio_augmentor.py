import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
import io
import base64
import random
import logging
from utils.audio_utils import generate_waveform

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AudioAugmentor:
    def __init__(self):
        self.operations = {
            "pitch_shift": self.pitch_shift,
            "time_stretch": self.time_stretch,
            "add_noise": self.add_noise,
            "reverse": self.reverse_audio,
            "volume_change": self.change_volume,
            "room_simulation": self.room_simulation,
            "compression": self.apply_compression,
            "chorus": self.apply_chorus,
            "echo": self.add_echo
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int, str]:
        try:
            if operation in self.operations:
                logger.debug(f"Applying operation: {operation}")
                augmented_audio, augmented_sr = self.operations[operation](audio_tensor.to(self.device), sr)
                waveform = generate_waveform(augmented_audio.cpu(), augmented_sr)
                return augmented_audio, augmented_sr, waveform
            return audio_tensor, sr, None
        except Exception as e:
            logger.error(f"Error in {operation}: {str(e)}")
            return audio_tensor, sr, None

    def pitch_shift(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            n_steps = random.uniform(-4, 4)
            logger.debug(f"Pitch shifting by {n_steps} steps")
            pitch_shifter = T.PitchShift(sr, n_steps)
            return pitch_shifter(audio_tensor), sr
        except Exception as e:
            logger.error(f"Error in pitch shift: {str(e)}")
            return audio_tensor, sr

    def time_stretch(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            rate = random.uniform(0.8, 1.2)
            logger.debug(f"Time stretching with rate {rate}")
            stretch = T.TimeStretch(n_freq=128, hop_length=512)
            return stretch(audio_tensor, rate), sr
        except Exception as e:
            logger.error(f"Error in time stretch: {str(e)}")
            return audio_tensor, sr

    def add_noise(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            noise = torch.randn_like(audio_tensor) * 0.005
            return torch.clamp(audio_tensor + noise, -1, 1), sr
        except Exception as e:
            logger.error(f"Error adding noise: {str(e)}")
            return audio_tensor, sr

    def reverse_audio(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            return torch.flip(audio_tensor, [-1]), sr
        except Exception as e:
            logger.error(f"Error reversing audio: {str(e)}")
            return audio_tensor, sr

    def change_volume(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            volume = random.uniform(0.5, 1.5)
            logger.debug(f"Changing volume by factor {volume}")
            return audio_tensor * volume, sr
        except Exception as e:
            logger.error(f"Error changing volume: {str(e)}")
            return audio_tensor, sr

    def room_simulation(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            # Create room impulse response using exponential decay
            ir_length = sr // 2
            ir = torch.exp(-torch.linspace(0, 5, ir_length)).to(self.device)
            ir = ir.unsqueeze(0)  # Add channel dimension
            
            # Apply convolution for room effect
            audio_tensor = F.convolve(audio_tensor, ir)
            return audio_tensor, sr
        except Exception as e:
            logger.error(f"Error in room simulation: {str(e)}")
            return audio_tensor, sr

    def apply_compression(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            threshold = 0.1
            ratio = 4.0
            logger.debug(f"Applying compression with threshold {threshold} and ratio {ratio}")
            
            magnitude = torch.abs(audio_tensor)
            mask = magnitude > threshold
            compressed = torch.where(
                mask,
                threshold + (magnitude - threshold) / ratio * torch.sign(audio_tensor),
                audio_tensor
            )
            return compressed, sr
        except Exception as e:
            logger.error(f"Error applying compression: {str(e)}")
            return audio_tensor, sr

    def apply_chorus(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            delays = [int(sr * d) for d in [0.02, 0.03, 0.04]]
            depths = [0.8, 0.6, 0.4]
            logger.debug(f"Applying chorus with {len(delays)} voices")
            
            chorus = torch.zeros_like(audio_tensor)
            for delay, depth in zip(delays, depths):
                pad = torch.zeros(audio_tensor.shape[0], delay).to(self.device)
                delayed = torch.cat([pad, audio_tensor[..., :-delay]], dim=-1)
                chorus += delayed * depth
            
            return (audio_tensor + chorus) / 2, sr
        except Exception as e:
            logger.error(f"Error applying chorus: {str(e)}")
            return audio_tensor, sr

    def add_echo(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            delay_time = 0.3  # seconds
            decay = 0.5
            delay_samples = int(sr * delay_time)
            logger.debug(f"Adding echo with {delay_time}s delay and {decay} decay")
            
            pad = torch.zeros(audio_tensor.shape[0], delay_samples).to(self.device)
            delayed = torch.cat([pad, audio_tensor[..., :-delay_samples]], dim=-1)
            echo = delayed * decay
            
            return audio_tensor + echo, sr
        except Exception as e:
            logger.error(f"Error adding echo: {str(e)}")
            return audio_tensor, sr