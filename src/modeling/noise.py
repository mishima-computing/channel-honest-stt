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
                
        # Preload buffers
        self._buffers = {}

    @staticmethod
    def _index_wavs(root):
        if not os.path.isdir(root): return []
        return sorted(glob.glob(os.path.join(root, '**', '*.wav'), recursive=True))

    def _load(self, path, sr):
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio.astype(np.float32)

    def preload(self, sr, duration_sec=300):
        # Preload 5 minutes of concatenated noise per category to speed up sampling
        print(f"Preloading {duration_sec}s of MUSAN noise per category...")
        target_len = int(sr * duration_sec)
        
        if 'music' in self.categories:
            buf = []
            while sum(len(x) for x in buf) < target_len:
                path = self.rng.choice(self._files['music'])
                buf.append(self._load(path, sr))
            self._buffers['music'] = np.concatenate(buf)
            
        if 'babble' in self.categories:
            # Generate babble mix
            buf = np.zeros(target_len, dtype=np.float32)
            lo, hi = self.babble_num_speakers
            n = min(self.rng.randint(lo, hi + 1), len(self._files['babble']))
            idx = self.rng.choice(len(self._files['babble']), size=n, replace=False)
            for i in idx:
                spk_buf = []
                while sum(len(x) for x in spk_buf) < target_len:
                    path = self.rng.choice(self._files['babble'])
                    spk_buf.append(self._load(path, sr))
                spk = np.concatenate(spk_buf)[:target_len]
                rms = np.sqrt(np.mean(spk ** 2))
                if rms > 0: spk = spk / rms
                buf += spk
            self._buffers['babble'] = buf
            
        print("MUSAN Preload complete.")

    def sample_noise(self, length, sr, category=None):
        if category is None: category = self.categories[self.rng.randint(len(self.categories))]
        
        if category in self._buffers:
            # Fast slice from preloaded buffer
            buf = self._buffers[category]
            start = self.rng.randint(0, len(buf) - length + 1)
            return buf[start:start+length]
        else:
            # Fallback to slow load
            if category == 'music': 
                path = self.rng.choice(self._files['music'])
                noise = self._load(path, sr)
            else:
                lo, hi = self.babble_num_speakers
                n = min(self.rng.randint(lo, hi + 1), len(self._files['babble']))
                idx = self.rng.choice(len(self._files['babble']), size=n, replace=False)
                noise = np.zeros(length, dtype=np.float32)
                for i in idx:
                    spk = self._load(self._files['babble'][i], sr)
                    if len(spk) < length: spk = np.tile(spk, int(np.ceil(length / len(spk))))
                    start = self.rng.randint(0, len(spk) - length + 1)
                    spk = spk[start:start+length]
                    rms = np.sqrt(np.mean(spk ** 2))
                    if rms > 0: spk = spk / rms
                    noise += spk
                return noise
                
            if len(noise) < length: noise = np.tile(noise, int(np.ceil(length / len(noise))))
            start = self.rng.randint(0, len(noise) - length + 1)
            return noise[start:start + length]

    def inject(self, signal, sr, snr_db, category=None):
        signal = np.asarray(signal, dtype=np.float32)
        noise = self.sample_noise(len(signal), sr, category=category)
        return add_noise(signal, noise, snr_db)
