import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sr=8000, n_mels=40, n_fft=256, hop_length=128):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_log_mel(self, y):
        """
        Extracts Log-Mel Filterbank energies from the audio signal.
        Returns a 2D array of shape (n_mels, t)
        """
        # Ensure floating point and normalize if needed
        if y.dtype != np.float32 and y.dtype != np.float64:
            y = y.astype(np.float32)
            
        melspec = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels,
            fmin=300.0,
            fmax=3400.0
        )
        # Convert to log scale (dB)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        return log_melspec

    def extract_vector(self, y):
        """
        Extracts a single fixed-length feature vector per utterance.
        For steady-state vowels, we simply average the Log-Mel features over time.
        """
        log_melspec = self.extract_log_mel(y)
        # Average over the time dimension (axis=1)
        mean_features = np.mean(log_melspec, axis=1)
        return mean_features

    def extract_dynamic(self, y, target_frames=50):
        """
        Extracts dynamic time-series features for consonants (keeps the time dimension).
        Pads or truncates to a fixed number of frames (target_frames) to allow batching.
        Returns a 2D array of shape (n_mels, target_frames)
        """
        log_melspec = self.extract_log_mel(y)
        n_mels, t = log_melspec.shape
        
        if t >= target_frames:
            # Truncate
            return log_melspec[:, :target_frames]
        else:
            # Pad with minimum value (silence)
            pad_width = target_frames - t
            min_val = np.min(log_melspec)
            padded = np.pad(log_melspec, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_val)
            return padded
