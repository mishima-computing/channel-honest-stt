import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor
from src.modeling.features import FeatureExtractor

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    print("Initializing CV Extractor...")
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    
    speakers = ['jvs001', 'jvs002']
    target_consonants = ['s', 'h', 'm', 'n']
    target_vowel = 'a'
    
    print(f"Extracting CV syllables (Consonants: {target_consonants}, Vowel: {target_vowel})...")
    X_clean, X_deg, y = extractor.extract_cv_syllables(speakers, target_consonants, target_vowel)
    
    print(f"Extracted {len(y)} CV syllables in total.")
    if len(y) == 0:
        print("No samples found. Check paths and mock data setup.")
        return
        
    feat_extractor = FeatureExtractor(sr=8000)
    
    # Let's plot one example per consonant
    plotted = set()
    fig, axes = plt.subplots(len(target_consonants), 2, figsize=(12, 3 * len(target_consonants)))
    
    for sig_clean, sig_deg, label in zip(X_clean, X_deg, y):
        if label not in plotted:
            idx = target_consonants.index(label)
            
            # Since sig_clean is 24kHz, let's just plot the Degraded features (8kHz)
            # because the model only sees the degraded features.
            feat_deg = feat_extractor.extract_dynamic(sig_deg, target_frames=40)
            
            # Plot Degraded Waveform
            ax1 = axes[idx, 0] if len(target_consonants) > 1 else axes[0]
            t = np.arange(len(sig_deg)) / 8000.0
            ax1.plot(t, sig_deg, color='blue')
            ax1.set_title(f"Degraded Waveform: /{label}{target_vowel}/")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            
            # Plot Degraded Log-Mel Spectrogram
            ax2 = axes[idx, 1] if len(target_consonants) > 1 else axes[1]
            im = ax2.imshow(feat_deg, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_title(f"Log-Mel Features (Phase 1): /{label}{target_vowel}/")
            ax2.set_xlabel("Frame Index")
            ax2.set_ylabel("Mel Bin (300-3400Hz)")
            fig.colorbar(im, ax=ax2, format='%+2.0f dB')
            
            plotted.add(label)
            
        if len(plotted) == len(target_consonants):
            break
            
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'cv_extraction_test.png')
    plt.savefig(out_path, dpi=150)
    print(f"Spectrograms saved to {out_path}")

if __name__ == '__main__':
    main()
