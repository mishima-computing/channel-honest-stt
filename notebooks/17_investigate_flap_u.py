import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    pipe = DegradationPipeline(target_sr=8000)
    
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    # 1. Extract clean features
    X_clean_c, y_c, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='consonant', cons_pre_ms=80.0
    )
    X_clean_v, y_v, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='vowel', vowel_dur_ms=80.0
    )
    
    X_clean_all = X_clean_c + X_clean_v
    y_all = y_c + y_v
    
    classes = [
        'Unvoiced Plosive', 'Voiced Plosive', 'Unvoiced Fricative', 'Nasal', 'Flap',
        '/a/', '/i/', '/u/', '/e/', '/o/'
    ]
    
    X_features = []
    for x_clean in X_clean_all:
        x_deg = pipe.process(x_clean, extractor.sr_orig, hpf_cutoff=500.0)
        feat = extract_log_mel(x_deg, 8000)
        X_features.append(feat)
        
    X_features = np.array(X_features)
    y_array = np.array(y_all)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_array, test_size=0.2, random_state=42, stratify=y_array
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    print("\n--- 10-Class Clean Accuracy Ranking ---")
    accs = {c: cm_norm[i, i] for i, c in enumerate(classes)}
    for c, acc in sorted(accs.items(), key=lambda item: item[1], reverse=True):
        print(f"{c}: {acc:.1%}")
        
    print("\n--- Flap Misclassification Distribution ---")
    flap_idx = classes.index('Flap')
    for i, c in enumerate(classes):
        if cm_norm[flap_idx, i] > 0.05:
            print(f"  -> {c}: {cm_norm[flap_idx, i]:.1%}")
            
    print("\n--- /u/ Misclassification Distribution ---")
    u_idx = classes.index('/u/')
    for i, c in enumerate(classes):
        if cm_norm[u_idx, i] > 0.05:
            print(f"  -> {c}: {cm_norm[u_idx, i]:.1%}")
            
    print("\n--- Voiced Plosive Misclassification Distribution ---")
    vp_idx = classes.index('Voiced Plosive')
    for i, c in enumerate(classes):
        if cm_norm[vp_idx, i] > 0.05:
            print(f"  -> {c}: {cm_norm[vp_idx, i]:.1%}")

if __name__ == '__main__':
    main()
