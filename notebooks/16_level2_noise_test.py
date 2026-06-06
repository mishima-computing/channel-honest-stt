import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor
from src.synthesizer.pipeline import DegradationPipeline
from src.modeling.noise import NoiseGenerator, add_noise

def extract_log_mel(audio, sr, n_mels=40):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=256, hop_length=64, n_mels=n_mels, fmin=0, fmax=4000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.flatten()

def plot_confusion_matrix(y_true, y_pred, classes, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='.1%')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    pipe = DegradationPipeline(target_sr=8000)
    noise_gen = NoiseGenerator(noise_type='pink', seed=42)
    
    # Use 10 speakers
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("Extracting clean 24kHz audio...")
    # 1. Extract clean consonants
    X_clean_c, y_c, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='consonant', cons_pre_ms=80.0
    )
    # 2. Extract clean vowels
    X_clean_v, y_v, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='vowel', vowel_dur_ms=80.0
    )
    
    X_clean_all = X_clean_c + X_clean_v
    y_all = y_c + y_v
    
    classes = [
        'Unvoiced Plosive', 'Voiced Plosive', 'Unvoiced Fricative', 'Nasal', 'Flap',
        '/a/', '/i/', '/u/', '/e/', '/o/'
    ]
    
    snr_levels = [None, 20.0, 10.0, 5.0, 0.0]
    
    class_accuracies = {c: [] for c in classes}
    overall_accuracies = []
    
    # Pre-generate a long noise array to avoid repeatedly generating it
    # Max length needed is 80ms * 24kHz = 1920 samples
    max_len = max(len(x) for x in X_clean_all)
    master_noise = noise_gen.generate(max_len * 100) # Plenty of noise to slice from
    
    for snr in snr_levels:
        snr_label = "Clean" if snr is None else f"{snr}dB"
        print(f"\n--- Testing SNR: {snr_label} ---")
        
        X_features = []
        for x_clean in X_clean_all:
            if snr is not None:
                # Add noise
                start = np.random.randint(0, len(master_noise) - len(x_clean))
                noise_slice = master_noise[start:start+len(x_clean)]
                x_mixed = add_noise(x_clean, noise_slice, snr)
            else:
                x_mixed = x_clean
                
            # Degrade
            x_deg = pipe.process(x_mixed, extractor.sr_orig, hpf_cutoff=500.0)
            
            # Extract feature
            feat = extract_log_mel(x_deg, 8000)
            X_features.append(feat)
            
        X_features = np.array(X_features)
        y_array = np.array(y_all)
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_array, test_size=0.2, random_state=42, stratify=y_array
        )
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42)
        clf.fit(X_train_s, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        overall_accuracies.append(acc)
        print(f"Overall Accuracy: {acc:.2%}")
        
        # Per-class accuracy
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        for i, c in enumerate(classes):
            class_accuracies[c].append(cm_norm[i, i])
            
        # Plot CM
        safe_snr = "clean" if snr is None else f"{snr}dB"
        cm_path = os.path.join(os.path.dirname(__file__), f'cm_noise_{safe_snr}.png')
        plot_confusion_matrix(y_test, y_pred, classes, f"Confusion Matrix (SNR: {snr_label})", cm_path)
        print(f"Saved CM to {cm_path}")

    # Plot SNR robustness curve
    plt.figure(figsize=(12, 8))
    x_vals = [30 if snr is None else snr for snr in snr_levels] # Use 30 for 'Clean'
    x_labels = ['Clean' if snr is None else f"{snr}dB" for snr in snr_levels]
    
    # Plot vowels (solid)
    vowels = ['/a/', '/i/', '/u/', '/e/', '/o/']
    for v in vowels:
        plt.plot(x_vals, class_accuracies[v], marker='o', linestyle='-', label=f"Vowel {v}")
        
    # Plot consonants (dashed)
    consonants = [c for c in classes if c not in vowels]
    for c in consonants:
        plt.plot(x_vals, class_accuracies[c], marker='x', linestyle='--', label=c)
        
    # Plot overall
    plt.plot(x_vals, overall_accuracies, marker='s', linestyle='-', color='black', linewidth=2, label="Overall")
    
    plt.title("SNR vs Accuracy (10-Class Baseline, Pink Noise)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.xticks(x_vals, x_labels)
    plt.ylim([0.0, 1.05])
    plt.gca().invert_xaxis() # High SNR to Low SNR
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    curve_path = os.path.join(os.path.dirname(__file__), 'snr_robustness_curve.png')
    plt.savefig(curve_path, dpi=150)
    print(f"Saved SNR Curve to {curve_path}")

if __name__ == '__main__':
    main()
