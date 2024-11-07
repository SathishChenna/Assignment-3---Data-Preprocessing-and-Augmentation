import torch
import matplotlib.pyplot as plt
import io
import base64
import logging

logger = logging.getLogger(__name__)

def generate_waveform(audio_tensor: torch.Tensor, sr: int) -> str:
    try:
        plt.clf()  # Clear the current figure
        plt.figure(figsize=(10, 2))
        
        # Ensure we're working with the correct shape
        if audio_tensor.dim() == 2:
            # If stereo, convert to mono by averaging channels
            waveform = audio_tensor.mean(dim=0)
        else:
            waveform = audio_tensor.squeeze()
        
        # Create time axis
        time_axis = torch.linspace(0, len(waveform)/sr, len(waveform))
        
        # Plot the waveform
        plt.plot(time_axis.numpy(), waveform.numpy())
        plt.title('Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close('all')  # Close all figures
        buf.seek(0)
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating waveform: {str(e)}")
        return ""