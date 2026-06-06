import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.dataset import VowelDatasetGenerator
from src.modeling.features import FeatureExtractor

def main():
    print("Generating Dataset (Phase 1 Degradation)...")
    generator = VowelDatasetGenerator()
    # 500 samples per class to keep it fast but statistically significant
    X_raw, y_raw = generator.generate_dataset(samples_per_class=500, seed=42)
    
    print("Extracting Log-Mel Features...")
    extractor = FeatureExtractor()
    X_features = []
    for sig in X_raw:
        feat = extractor.extract_vector(sig)
        X_features.append(feat)
        
    X = np.array(X_features)
    y = np.array(y_raw)
    
    print(f"Dataset Shape: X={X.shape}, y={y.shape}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Logistic Regression Model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    # Generate Confusion Matrix
    labels = ['/i/', '/e/', '/a/', '/o/', '/u/', '/e-i/', '/u-o/']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Phase 1 (Clean Channel) Confusion Matrix\nAccuracy: {acc * 100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix_phase1.png')
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")

if __name__ == '__main__':
    main()
