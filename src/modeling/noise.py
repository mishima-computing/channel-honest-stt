import numpy as np

class NoiseGenerator:
    def __init__(self, noise_type='pink', seed=42):
        self.noise_type = noise_type
        self.rng = np.random.RandomState(seed)

    def generate(self, length):
        """Generates noise of the specified length."""
        if self.noise_type == 'white':
            return self.rng.randn(length).astype(np.float32)
        elif self.noise_type == 'pink':
            # Simple approximation of pink noise in frequency domain
            white = self.rng.randn(length).astype(np.float32)
            X = np.fft.rfft(white)
            S = np.fft.rfftfreq(length)
            S[0] = S[1] # Avoid divide by zero
            X /= np.sqrt(S) # 1/f power spectrum
            pink = np.fft.irfft(X, n=length)
            return pink.astype(np.float32)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def inject(self, signal, snr_db):
        noise = self.generate(len(signal))
        return add_noise(signal, noise, snr_db)

def add_noise(signal, noise, snr_db):
    """
    Adds noise to a signal at a specified SNR.
    The SNR is calculated based on the RMS of the entire provided signal.
    """
    if len(noise) < len(signal):
        # Tile noise if it's too short
        noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
    # Randomly slice noise if it's longer
    if len(noise) > len(signal):
        start = np.random.randint(0, len(noise) - len(signal) + 1)
        noise = noise[start:start+len(signal)]
        
    signal_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    
    if noise_rms == 0 or signal_rms == 0:
        return signal
        
    # SNR = 20 * log10(RMS_s / RMS_n)
    # RMS_n = RMS_s / (10^(SNR/20))
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return signal + scaled_noise

import os
import glob

class MusanInjector:
    def __init__(self, musan_root, categories=('babble', 'music'), babble_num_speakers=(3, 7), seed=42):
        self.musan_root = musan_root
        self.categories = tuple(categories)
        self.babble_num_speakers = babble_num_speakers
        self.rng = np.random.RandomState(seed)
        self._files = {
            'music': self._index_wavs(os.path.join(musan_root, 'music')),
            'babble': self._index_wavs(os.path.join(musan_root, 'speech')),
        }
        for cat in self.categories:
            if cat not in self._files:
                raise ValueError(f"Unknown MUSAN category: {cat!r}")
            if not self._files[cat]:
                pass # Delay check until used

    @staticmethod
    def _index_wavs(root):
        if not os.path.isdir(root): return []
        return sorted(glob.glob(os.path.join(root, '**', '*.wav'), recursive=True))

    def _load(self, path, sr):
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio.astype(np.float32)

    def _fit_length(self, noise, length):
        if len(noise) < length: noise = np.tile(noise, int(np.ceil(length / len(noise))))
        if len(noise) > length:
            start = self.rng.randint(0, len(noise) - length + 1)
            noise = noise[start:start + length]
        return noise

    def _make_music(self, length, sr):
        path = self._files['music'][self.rng.randint(len(self._files['music']))]
        return self._fit_length(self._load(path, sr), length)

    def _make_babble(self, length, sr):
        lo, hi = self.babble_num_speakers
        files = self._files['babble']
        n = min(self.rng.randint(lo, hi + 1), len(files))
        idx = self.rng.choice(len(files), size=n, replace=False)
        babble = np.zeros(length, dtype=np.float32)
        for i in idx:
            spk = self._fit_length(self._load(files[i], sr), length)
            rms = np.sqrt(np.mean(spk ** 2))
            if rms > 0: spk = spk / rms
            babble += spk
        return babble

    def sample_noise(self, length, sr, category=None):
        if category is None: category = self.categories[self.rng.randint(len(self.categories))]
        if not self._files[category]:
             self._files = {
                'music': self._index_wavs(os.path.join(self.musan_root, 'music')),
                'babble': self._index_wavs(os.path.join(self.musan_root, 'speech')),
             }
        if not self._files[category]: raise FileNotFoundError(f"No files for {category}")
        if category == 'music': return self._make_music(length, sr)
        if category == 'babble': return self._make_babble(length, sr)

    def inject(self, signal, sr, snr_db, category=None):
        signal = np.asarray(signal, dtype=np.float32)
        noise = self.sample_noise(len(signal), sr, category=category)
        return add_noise(signal, noise, snr_db)
