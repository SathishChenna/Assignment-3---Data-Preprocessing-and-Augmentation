import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
import io
import base64

class AudioAugmentor:
    def __init__(self):
        self.operations = {
            "pitch_shift": self.pitch_shift,
            "time_stretch": self.time_stretch,
            "add_noise": self.add_noise,
            "reverse": self.reverse_audio,
            "volume_change": self.change_volume
        }
    
    def get_available_operations(self):
        return list(self.operations.keys())
    
    def apply_operation(self, operation: str, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int, str]:
        if operation in self.operations:
            augmented_audio, augmented_sr = self.operations[operation](audio_data, sr)
            # Generate time domain visualization
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
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def pitch_shift(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        # Shift pitch up or down by random steps
        steps = random.uniform(-4, 4)
        return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=steps), sr

    def time_stretch(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        # Stretch or compress the audio by a random rate
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio_data, rate=rate), sr

    def add_noise(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        noise = np.random.normal(0, 0.005, audio_data.shape)
        return audio_data + noise, sr

    def reverse_audio(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        return np.flip(audio_data), sr

    def change_volume(self, audio_data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        # Random volume change between 0.5x and 1.5x
        volume = random.uniform(0.5, 1.5)
        return audio_data * volume, sr 