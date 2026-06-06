import numpy as np
import scipy.signal as signal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.synthesizer.formant import FormantSynthesizer
from src.synthesizer.pipeline import DegradationPipeline

class VowelDatasetGenerator:
    def __init__(self, sr_synth=48000, sr_deg=8000):
        self.synth = FormantSynthesizer(sr_synth)
        self.pipe = DegradationPipeline(sr_deg)
        self.sr_synth = sr_synth
        self.sr_deg = sr_deg
        
        self.vowel_targets = {
            '/i/': [(300, 50), (2300, 100), (3000, 120)],
            '/e/': [(450, 60), (2000, 100), (2800, 120)],
            '/a/': [(700, 80), (1200, 100), (2600, 120)],
            '/o/': [(500, 60), (900, 100), (2400, 120)],
            '/u/': [(350, 50), (1300, 100), (2500, 120)],
            '/e-i/': [(370, 55), (2150, 100), (2900, 120)],
            '/u-o/': [(420, 55), (1100, 100), (2450, 120)]
        }
        
    def generate_randomized_formants(self, target_formants, jitter_pct=0.05):
        randomized = []
        for f, bw in target_formants:
            # Randomize frequency by ±jitter_pct
            f_rand = f * np.random.uniform(1.0 - jitter_pct, 1.0 + jitter_pct)
            randomized.append((f_rand, bw))
        return randomized

    def generate_dataset(self, samples_per_class=1000, seed=42):
        np.random.seed(seed)
        X = []
        y = []
        
        for class_idx, (name, targets) in enumerate(self.vowel_targets.items()):
            for _ in range(samples_per_class):
                # Randomize F0 between 100Hz and 250Hz
                f0 = np.random.uniform(100.0, 250.0)
                
                # Randomize duration between 0.1s and 0.3s
                duration = np.random.uniform(0.1, 0.3)
                
                # Randomize formants (2% jitter to prevent overlap between adjacent wells like /e/ and /e-i/)
                formants = self.generate_randomized_formants(targets, 0.02)
                
                # Synthesize
                t, y_clean = self.synth.synthesize_vowel(f0, formants, duration)
                
                # Degrade (Phase 1: resample, u-law, hpf, tanh)
                y_deg = self.pipe.process(y_clean, self.sr_synth, hpf_cutoff=500.0)
                
                X.append(y_deg)
                y.append(name)
                
        return X, y
