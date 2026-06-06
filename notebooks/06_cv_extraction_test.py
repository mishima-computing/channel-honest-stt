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
    # Use fixed length extraction: 60ms before vowel, 40ms after vowel = 100ms total
    X_clean, X_deg, y, meta = extractor.extract_cv_syllables_vowel_anchor(speakers, target_consonants, target_vowel, pre_vowel_ms=60.0, post_vowel_ms=40.0)
    
    print(f"Extracted {len(y)} CV syllables in total.")
    if len(y) == 0:
        print("No samples found. Check paths and dataset setup.")
        return
        
    feat_extractor = FeatureExtractor(sr=8000)
    
    # Plot one example per consonant
    plotted = set()
    fig, axes = plt.subplots(len(target_consonants), 2, figsize=(12, 3 * len(target_consonants)))
    
    for sig_clean, sig_deg, label, m in zip(X_clean, X_deg, y, meta):
        if label not in plotted:
            idx = target_consonants.index(label)
            
            # Since sig_deg is exactly 100ms, extract_log_mel naturally returns a fixed-size tensor (e.g., 40x7)
            # No zero-padding needed!
            feat_deg = feat_extractor.extract_log_mel(sig_deg)
            
            # Plot Degraded Waveform
            ax1 = axes[idx, 0] if len(target_consonants) > 1 else axes[0]
            t = np.arange(len(sig_deg)) / 8000.0
            ax1.plot(t, sig_deg, color='blue', alpha=0.7)
            
            # Draw lab boundaries
            ax1.axvline(x=m['cons_start_rel'], color='r', linestyle='--', label='Consonant Start (Julius)')
            ax1.axvline(x=m['vowel_start_rel'], color='g', linestyle='-', linewidth=2, label='Vowel Anchor')
            
            ax1.set_title(f"Waveform (100ms Fixed): /{label}{target_vowel}/ ({m['speaker']})")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            if idx == 0:
                ax1.legend(loc='upper left', fontsize='small')
            
            # Plot Degraded Log-Mel Spectrogram
            ax2 = axes[idx, 1] if len(target_consonants) > 1 else axes[1]
            im = ax2.imshow(feat_deg, aspect='auto', origin='lower', cmap='viridis')
            
            # hop_length = 128 at 8000Hz (from FeatureExtractor defaults)
            hop_sec = 128.0 / 8000.0
            ax2.axvline(x=m['cons_start_rel'] / hop_sec, color='r', linestyle='--')
            ax2.axvline(x=m['vowel_start_rel'] / hop_sec, color='g', linestyle='-', linewidth=2)
            
            ax2.set_title(f"Log-Mel (No Pad): /{label}{target_vowel}/ Shape={feat_deg.shape}")
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
