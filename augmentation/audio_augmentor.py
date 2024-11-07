import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
import io
import base64
from scipy import signal

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
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int, str]:
        if operation in self.operations:
            augmented_audio, augmented_sr = self.operations[operation](audio_data, sr)
            waveform = self.generate_waveform(augmented_audio, augmented_sr)
            return augmented_audio, augmented_sr, waveform
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

    def pitch_shift(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        steps = random.uniform(-4, 4)
        return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=steps), sr

    def time_stretch(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio_data, rate=rate), sr

    def add_noise(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        noise = np.random.normal(0, 0.005, audio_data.shape)
        return audio_data + noise, sr

    def reverse_audio(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        return np.flip(audio_data), sr

    def change_volume(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        volume = random.uniform(0.5, 1.5)
        return audio_data * volume, sr

    def room_simulation(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        room_ir = np.exp(-np.linspace(0, 2, sr//2))
        return signal.convolve(audio_data, room_ir, mode='same'), sr

    def apply_compression(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        threshold = 0.1
        ratio = 4.0
        audio_data = np.where(
            np.abs(audio_data) > threshold,
            threshold + (np.abs(audio_data) - threshold) / ratio * np.sign(audio_data),
            audio_data
        )
        return audio_data, sr

    def apply_chorus(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        delays = [int(sr * d) for d in [0.02, 0.03, 0.04]]
        depths = [0.8, 0.6, 0.4]
        chorus = np.zeros_like(audio_data)
        for delay, depth in zip(delays, depths):
            delayed = np.pad(audio_data, (delay, 0))[:len(audio_data)]
            chorus += delayed * depth
        return (audio_data + chorus) / 2, sr

    def add_echo(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        delay_time = 0.3  # seconds
        decay = 0.5
        delay_samples = int(sr * delay_time)
        echo = np.zeros_like(audio_data)
        echo[delay_samples:] = audio_data[:-delay_samples] * decay
        return audio_data + echo, sr