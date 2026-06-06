import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor
from src.modeling.features import FeatureExtractor

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    
    # Use 10 speakers
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("Extracting Track A Consonant-only features (-60ms to 0ms)...")
    X_deg, y_class, meta = extractor.extract_diagnostic_features(speakers, target_type='consonant', cons_pre_ms=60.0)
    
    print(f"Extracted {len(y_class)} valid samples.")
    
    # Feature extraction
    feat_extractor = FeatureExtractor(sr=8000)
    X_features = []
    
    for sig_deg in X_deg:
        feat = feat_extractor.extract_log_mel(sig_deg)
        X_features.append(feat.flatten())
        
    X_features = np.array(X_features)
    y_class = np.array(y_class)
    
    print(f"Feature matrix shape: {X_features.shape}")
    
    class_order = [
        'Unvoiced Plosive', 'Voiced Plosive', 
        'Unvoiced Fricative', 'Unvoiced Affricate', 'Voiced Affricate/Fricative', 
        'Nasal', 'Flap'
    ]
    unique_classes = np.unique(y_class)
    present_classes = [c for c in class_order if c in unique_classes]
    
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred_all = np.zeros_like(y_class)
    
    print("Training Logistic Regression on Consonants...")
    for train_idx, test_idx in skf.split(X_features, y_class):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y_class[train_idx], y_class[test_idx]
        clf.fit(X_train, y_train)
        y_pred_all[test_idx] = clf.predict(X_test)
        
    print("\nClassification Report (Consonants):")
    print(classification_report(y_class, y_pred_all, labels=present_classes))
    
    cm = confusion_matrix(y_class, y_pred_all, labels=present_classes)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=present_classes, yticklabels=present_classes)
    plt.title("Diagnostic Step 3: Consonant-Only Confusion Matrix (No Vowel)")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    out_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix_consonant_only.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Confusion matrix saved to {out_path}")

if __name__ == '__main__':
    main()
