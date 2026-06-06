import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor
from src.synthesizer.pipeline import DegradationPipeline
from src.modeling.noise import MusanInjector

def extract_log_mel(audio, sr, n_mels=40):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=256, hop_length=64, n_mels=n_mels, fmin=0, fmax=4000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.flatten()

def process_and_eval(X_clean, y_labels, target_classes, title_prefix, snr, category, pipe, musan, extractor):
    X_features = []
    
    desc = f"Extracting {title_prefix} (SNR={snr if snr is not None else 'Clean'}, Cat={category})"
    for x_clean in tqdm(X_clean, desc=desc, leave=False):
        if snr is not None:
            x_mixed = musan.inject(x_clean, extractor.sr_orig, snr, category=category)
        else:
            x_mixed = x_clean
            
        x_deg = pipe.process(x_mixed, extractor.sr_orig, hpf_cutoff=500.0)
        feat = extract_log_mel(x_deg, 8000)
        X_features.append(feat)
        
    X_features = np.array(X_features)
    y_array = np.array(y_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_array, test_size=0.2, random_state=42, stratify=y_array
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42)
    clf.fit(X_train_s, y_train)
    
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred, labels=target_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    total_test = len(y_test)
    correct_test = int(acc * total_test)
    
    return acc, cm_norm, total_test, correct_test

def run_evaluation(category, snr_levels, cons_classes, vowel_classes, X_c_raw, y_c, X_v_raw, y_v, pipe, musan, extractor):
    class_accuracies = {c: [] for c in cons_classes + vowel_classes}
    overall_accuracies = []
    
    for snr in snr_levels:
        acc_c, cm_c, total_c, correct_c = process_and_eval(X_c_raw, y_c, cons_classes, "Consonants", snr, category, pipe, musan, extractor)
        acc_v, cm_v, total_v, correct_v = process_and_eval(X_v_raw, y_v, vowel_classes, "Vowels", snr, category, pipe, musan, extractor)
        
        overall_acc = (correct_c + correct_v) / (total_c + total_v)
        overall_accuracies.append(overall_acc)
        
        for i, c in enumerate(cons_classes):
            class_accuracies[c].append(cm_c[i, i])
        for i, c in enumerate(vowel_classes):
            class_accuracies[c].append(cm_v[i, i])
            
    return overall_accuracies, class_accuracies

def plot_curve(category, snr_levels, overall_accuracies, class_accuracies, cons_classes, vowel_classes):
    plt.figure(figsize=(12, 8))
    x_vals = [30 if snr is None else snr for snr in snr_levels]
    x_labels = ['Clean' if snr is None else f"{snr}dB" for snr in snr_levels]
    
    for v in vowel_classes:
        plt.plot(x_vals, class_accuracies[v], marker='o', linestyle='-', label=f"Vowel {v}")
    for c in cons_classes:
        plt.plot(x_vals, class_accuracies[c], marker='x', linestyle='--', label=c)
        
    plt.plot(x_vals, overall_accuracies, marker='s', linestyle='-', color='black', linewidth=2, label="Overall")
    
    plt.title(f"MUSAN {category.capitalize()} Noise - Dual Window True Robustness (Matched)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.xticks(x_vals, x_labels)
    plt.ylim([0.0, 1.05])
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    curve_path = os.path.join(os.path.dirname(__file__), f'musan_robustness_curve_{category}.png')
    plt.savefig(curve_path, dpi=150)
    print(f"\nSaved MUSAN SNR Curve to {curve_path}")

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    musan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/musan'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    pipe = DegradationPipeline(target_sr=8000)
    musan = MusanInjector(musan_root=musan_root)
    musan.preload(sr=24000, duration_sec=60)
    
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("Extracting clean 24kHz audio...")
    X_c_raw, y_c, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='consonant', cons_pre_ms=20.0
    )
    X_v_raw, y_v, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='vowel', vowel_dur_ms=80.0
    )
    
    cons_classes = ['Unvoiced Plosive', 'Voiced Plosive', 'Unvoiced Fricative', 'Nasal']
    vowel_classes = ['/a/', '/i/', '/u/', '/e/', '/o/']
    
    snr_levels = [None, 20.0, 10.0, 5.0, 0.0]
    
    print("\n\n=== Evaluating MUSAN: BABBLE ===")
    overall_b, class_b = run_evaluation('babble', snr_levels, cons_classes, vowel_classes, X_c_raw, y_c, X_v_raw, y_v, pipe, musan, extractor)
    plot_curve('babble', snr_levels, overall_b, class_b, cons_classes, vowel_classes)
    
    print("\n\n=== Evaluating MUSAN: MUSIC ===")
    overall_m, class_m = run_evaluation('music', snr_levels, cons_classes, vowel_classes, X_c_raw, y_c, X_v_raw, y_v, pipe, musan, extractor)
    plot_curve('music', snr_levels, overall_m, class_m, cons_classes, vowel_classes)

if __name__ == '__main__':
    main()
