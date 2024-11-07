import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
import io
import base64
from utils.audio_utils import generate_waveform

class AudioPreprocessor:
    def __init__(self):
        self.operations = {
            "normalize": self.normalize_audio,
            "noise_reduction": self.reduce_noise,
            "trim_silence": self.trim_silence,
            "resample": self.resample_audio,
            "change_speed": self.change_speed,
            "remove_hum": self.remove_hum
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize transforms
        self.griffin_lim = T.GriffinLim(n_fft=2048)
        self.spec_transform = T.Spectrogram(n_fft=2048)
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int, str]:
        try:
            if operation in self.operations:
                processed_audio, processed_sr = self.operations[operation](audio_tensor.to(self.device), sr)
                waveform = generate_waveform(processed_audio.cpu(), processed_sr)
                return processed_audio, processed_sr, waveform
            return audio_tensor, sr, None
        except Exception as e:
            print(f"Error in {operation}: {str(e)}")
            return audio_tensor, sr, None

    def normalize_audio(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        return F.normalize(audio_tensor), sr

    def reduce_noise(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            # Convert to spectrogram
            spec = self.spec_transform(audio_tensor)
            # Estimate noise from first few frames
            noise_estimate = torch.mean(spec[:, :, :10], dim=2, keepdim=True)
            # Subtract noise
            spec_sub = torch.clamp(spec - noise_estimate, min=0)
            # Convert back to audio using Griffin-Lim
            denoised = self.griffin_lim(spec_sub)
            return denoised.unsqueeze(0), sr
        except Exception as e:
            print(f"Error in noise reduction: {str(e)}")
            return audio_tensor, sr

    def trim_silence(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            # Compute energy
            energy = torch.abs(audio_tensor).mean(dim=0)
            threshold = energy.mean() * 0.1
            mask = energy > threshold
            # Ensure we keep at least some audio
            if mask.sum() == 0:
                return audio_tensor, sr
            return audio_tensor[:, mask], sr
        except Exception as e:
            print(f"Error in trim silence: {str(e)}")
            return audio_tensor, sr

    def resample_audio(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            target_sr = 22050
            resampler = T.Resample(sr, target_sr)
            return resampler(audio_tensor), target_sr
        except Exception as e:
            print(f"Error in resampling: {str(e)}")
            return audio_tensor, sr

    def change_speed(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            # Using time stretch
            stretch = T.TimeStretch(n_freq=128, hop_length=512)
            return stretch(audio_tensor, rate=1.5), sr
        except Exception as e:
            print(f"Error in speed change: {str(e)}")
            return audio_tensor, sr

    def remove_hum(self, audio_tensor: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        try:
            # Using notch filter to remove 50/60 Hz hum
            notch_50 = T.Notch(sr, freq=50, q=30)
            notch_60 = T.Notch(sr, freq=60, q=30)
            audio_tensor = notch_50(audio_tensor)
            audio_tensor = notch_60(audio_tensor)
            return audio_tensor, sr
        except Exception as e:
            print(f"Error in hum removal: {str(e)}")
            return audio_tensor, sr

    @staticmethod
    def tensor_to_bytes(audio_tensor: torch.Tensor, sr: int) -> bytes:
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.cpu(), sr, format='wav')
        return buffer.getvalue()