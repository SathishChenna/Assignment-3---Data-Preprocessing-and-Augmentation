import librosa
import numpy as np
import soundfile as sf
import io
import matplotlib.pyplot as plt
import base64
from scipy.signal import iirnotch, filtfilt

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
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int, str]:
        if operation in self.operations:
            processed_audio, processed_sr = self.operations[operation](audio_data, sr)
            waveform = self.generate_waveform(processed_audio, processed_sr)
            return processed_audio, processed_sr, waveform
        return audio_data, sr, None

    def generate_waveform(self, audio_data: np.ndarray, sr: int) -> str:
        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, len(audio_data)/sr, len(audio_data)), audio_data)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def normalize_audio(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        return librosa.util.normalize(audio_data), sr

    def reduce_noise(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        return librosa.effects.preemphasis(audio_data), sr

    def trim_silence(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=20)
        return trimmed_audio, sr

    def resample_audio(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        target_sr = 22050
        resampled_audio = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        return resampled_audio, target_sr

    def change_speed(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        return librosa.effects.time_stretch(audio_data, rate=1.5), sr

    def remove_hum(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        # Apply notch filter to remove power line hum (50/60 Hz)
        nyquist = sr / 2
        quality_factor = 30.0
        
        # 50 Hz filter
        b_50, a_50 = iirnotch(50 / nyquist, quality_factor)
        y_50 = filtfilt(b_50, a_50, audio_data)
        
        # 60 Hz filter
        b_60, a_60 = iirnotch(60 / nyquist, quality_factor)
        y_60 = filtfilt(b_60, a_60, y_50)
        
        return y_60, sr

    @staticmethod
    def array_to_bytes(audio_data: np.ndarray, sr: int) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        return buffer.getvalue()