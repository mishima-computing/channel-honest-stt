import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor
from src.synthesizer.pipeline import DegradationPipeline
import librosa

def extract_log_mel(audio, sr, n_mels=40):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=256, hop_length=64, n_mels=n_mels, fmin=0, fmax=4000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.flatten()

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    # We will use cons_pre_ms=30.0 for all diagnostic classes so they have the same feature dimension.
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    pipe = DegradationPipeline(target_sr=8000)
    
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("Extracting features with 30ms window...")
    # Extract ALL consonants but we'll filter only Flap and Voiced Plosive
    X_clean_c, y_c, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='consonant', cons_pre_ms=30.0
    )
    # Extract Vowels but with 30ms pre-window instead of post-window?
    # Wait, the diagnostic for vowels usually uses [0, +80ms]. 
    # If we use [-30ms, 0ms] for vowels, we are extracting the END of the preceding consonant!
    # Ah! We cannot compare Flap [-30, 0] with Vowel [0, +30]. 
    # Or can we? The classifier just takes a fixed-length vector. 
    # For Flap, the feature is the 30ms BEFORE the vowel anchor.
    # For Vowel, the feature is the 30ms AFTER the vowel anchor.
    # Since they are different time segments relative to the anchor, we can just extract Vowel with [0, +30ms].
    X_clean_v, y_v, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='vowel', vowel_dur_ms=30.0
    )
    
    target_classes = ['Flap', 'Voiced Plosive', '/a/', '/i/', '/u/']
    
    X_filtered = []
    y_filtered = []
    
    # Filter and degrade consonants
    for x, y in zip(X_clean_c, y_c):
        if y in target_classes:
            x_deg = pipe.process(x, extractor.sr_orig, hpf_cutoff=500.0)
            feat = extract_log_mel(x_deg, 8000)
            X_filtered.append(feat)
            y_filtered.append(y)
            
    # Filter and degrade vowels
    for x, y in zip(X_clean_v, y_v):
        if y in target_classes:
            x_deg = pipe.process(x, extractor.sr_orig, hpf_cutoff=500.0)
            feat = extract_log_mel(x_deg, 8000)
            X_filtered.append(feat)
            y_filtered.append(y)
            
    X_features = np.array(X_filtered)
    y_array = np.array(y_filtered)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_array, test_size=0.2, random_state=42, stratify=y_array
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train_s, y_train)
    
    y_pred = clf.predict(X_test_s)
    
    cm = confusion_matrix(y_test, y_pred, labels=target_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    print("\n--- 30ms Window Sub-Classification ---")
    for i, c in enumerate(target_classes):
        print(f"{c}: {cm_norm[i, i]:.1%}")
        
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=target_classes)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='.1%')
    ax.set_title("Flap Diagnostic CM (30ms Window)")
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), 'flap_diagnostic_cm.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved CM to {out_path}")

if __name__ == '__main__':
    main()
