import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import librosa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor

def extract_log_mel(audio, sr, n_mels=40):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=256, hop_length=64, n_mels=n_mels, fmin=0, fmax=4000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.flatten()

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("Extracting unified 60ms [-20, +40] features for all 9 classes...")
    # target_type='all' extracts both consonants and vowels using their respective anchors
    X_deg_raw, y_labels, meta = extractor.extract_unified_features(speakers, target_type='all', pre_anchor_ms=20.0, post_anchor_ms=40.0)
    
    print(f"Extracted {len(X_deg_raw)} total samples.")
    
    X_features = []
    y_filtered = []
    
    for i in tqdm(range(len(X_deg_raw))):
        feat = extract_log_mel(X_deg_raw[i], 8000)
        X_features.append(feat)
        y_filtered.append(y_labels[i])
        
    X_features = np.array(X_features)
    y_array = np.array(y_filtered)
    
    target_classes = [
        'Unvoiced Plosive', 'Voiced Plosive', 'Unvoiced Fricative', 'Nasal',
        '/a/', '/i/', '/u/', '/e/', '/o/'
    ]
    
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
    
    acc = np.mean(y_test == y_pred)
    print(f"\nUnified Window Baseline Accuracy: {acc:.2%}")
    for i, c in enumerate(target_classes):
        print(f"{c}: {cm_norm[i, i]:.1%}")
        
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=target_classes)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='.1%')
    ax.set_title(f"Unified Window (60ms) - Clean Baseline (Acc: {acc:.2%})")
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), 'cm_unified_baseline.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved CM to {out_path}")

if __name__ == '__main__':
    main()
