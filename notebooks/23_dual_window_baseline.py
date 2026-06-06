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

def train_and_eval(X_raw, y_labels, target_classes, title_prefix):
    X_features = []
    for x in tqdm(X_raw, desc=f"Extracting Mels ({title_prefix})"):
        X_features.append(extract_log_mel(x, 8000))
        
    X_features = np.array(X_features)
    y_array = np.array(y_labels)
    
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
    total_test = len(y_test)
    correct_test = int(acc * total_test)
    
    print(f"--- {title_prefix} ---")
    print(f"Overall {title_prefix} Accuracy: {acc:.2%}")
    for i, c in enumerate(target_classes):
        print(f"{c}: {cm_norm[i, i]:.1%}")
    
    return acc, cm_norm, total_test, correct_test

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("1. Extracting Consonants (Dual Window: [-20ms, +40ms] anchored to burst/label)...")
    X_c_raw, y_c, _ = extractor.extract_dual_window_features(speakers, target_type='consonant')
    
    print("2. Extracting Vowels (Dual Window: [0ms, +80ms] anchored to vowel start)...")
    X_v_raw, y_v, _ = extractor.extract_clean_diagnostic_features(speakers, target_type='vowel', vowel_dur_ms=80.0)
    
    cons_classes = ['Unvoiced Plosive', 'Voiced Plosive', 'Unvoiced Fricative', 'Nasal']
    vowel_classes = ['/a/', '/i/', '/u/', '/e/', '/o/']
    
    acc_c, cm_c, total_c, correct_c = train_and_eval(X_c_raw, y_c, cons_classes, "Consonants")
    acc_v, cm_v, total_v, correct_v = train_and_eval(X_v_raw, y_v, vowel_classes, "Vowels")
    
    overall_acc = (correct_c + correct_v) / (total_c + total_v)
    
    print(f"\n--- Dual Window True Baseline ---")
    print(f"Consonant Accuracy (4 classes): {acc_c:.2%}")
    print(f"Vowel Accuracy (5 classes):     {acc_v:.2%}")
    print(f"Overall Weighted Accuracy:      {overall_acc:.2%}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    disp_c = ConfusionMatrixDisplay(confusion_matrix=cm_c, display_labels=cons_classes)
    disp_c.plot(cmap='Blues', ax=axes[0], xticks_rotation=45, values_format='.1%')
    axes[0].set_title(f"Consonants ([-20, +40] anchor)\nAcc: {acc_c:.2%}")
    
    disp_v = ConfusionMatrixDisplay(confusion_matrix=cm_v, display_labels=vowel_classes)
    disp_v.plot(cmap='Reds', ax=axes[1], xticks_rotation=45, values_format='.1%')
    axes[1].set_title(f"Vowels ([0, +80] anchor)\nAcc: {acc_v:.2%}")
    
    fig.suptitle(f"True Dual Window Baseline (Overall Accuracy: {overall_acc:.2%})", fontsize=16)
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), 'cm_true_dual_baseline.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved CM to {out_path}")

if __name__ == '__main__':
    main()
