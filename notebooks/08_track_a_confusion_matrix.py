import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor
from src.modeling.features import FeatureExtractor

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    
    # We will use the first 10 speakers for the confusion matrix to get a good amount of data
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("Extracting Track A 12-class features (Vowel-Anchor, 100ms)...")
    X_deg, y_class, meta = extractor.extract_track_a_features(speakers, pre_vowel_ms=60.0, post_vowel_ms=40.0)
    
    print(f"Extracted {len(y_class)} valid samples after context filtering.")
    if len(y_class) == 0:
        print("No samples extracted!")
        return

    # Count classes
    unique_classes, counts = np.unique(y_class, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        print(f"  {cls}: {count}")

    # Feature extraction
    feat_extractor = FeatureExtractor(sr=8000)
    X_features = []
    
    for sig_deg in X_deg:
        # extract_log_mel returns (n_mels, frames). Since sig_deg is exactly 100ms (800 samples),
        # frames will be exactly 7.
        feat = feat_extractor.extract_log_mel(sig_deg)
        # Flatten to 1D vector (40 * 7 = 280 dimensions)
        X_features.append(feat.flatten())
        
    X_features = np.array(X_features)
    y_class = np.array(y_class)
    
    print(f"Feature matrix shape: {X_features.shape}")
    
    # Target Class Order for Confusion Matrix
    class_order = [
        '/a/', '/i/', '/u/', '/e/', '/o/',
        'Unvoiced Plosive', 'Voiced Plosive', 
        'Unvoiced Fricative', 'Unvoiced Affricate', 'Voiced Affricate/Fricative', 
        'Nasal', 'Flap'
    ]
    
    # Filter classes to those actually present
    present_classes = [c for c in class_order if c in unique_classes]
    
    # Classification with Logistic Regression
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred_all = np.zeros_like(y_class)
    
    print("Training and evaluating Logistic Regression...")
    for train_idx, test_idx in skf.split(X_features, y_class):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y_class[train_idx], y_class[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred_all[test_idx] = clf.predict(X_test)
        
    print("\nClassification Report:")
    print(classification_report(y_class, y_pred_all, labels=present_classes))
    
    # Confusion Matrix
    cm = confusion_matrix(y_class, y_pred_all, labels=present_classes)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=present_classes, yticklabels=present_classes)
    plt.title("Track A Confusion Matrix (12 Classes) - JVS Real Speech Level 1")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # Save the plot
    out_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix_track_a.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Confusion matrix saved to {out_path}")

if __name__ == '__main__':
    main()
