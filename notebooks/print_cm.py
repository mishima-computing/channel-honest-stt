import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib.util
spec = importlib.util.spec_from_file_location("mod", "notebooks/15_investigate_zj_split.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
extract_zj_split_80ms = mod.extract_zj_split_80ms
JVSCVExtractor = mod.JVSCVExtractor
FeatureExtractor = mod.FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
speakers = [f'jvs{i:03d}' for i in range(1, 11)]

X_deg, y_class = extract_zj_split_80ms(extractor, speakers)
feat_extractor = FeatureExtractor(sr=8000)
X_features = np.array([feat_extractor.extract_log_mel(sig).flatten() for sig in X_deg])
y_class = np.array(y_class)

class_order = [
    'Unvoiced Plosive', 'Voiced Plosive', 
    'Unvoiced Fricative',
    'Voiced Affricate [dz]', 'Voiced Fricative [z]'
]

clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_all = np.zeros_like(y_class)

for train_idx, test_idx in skf.split(X_features, y_class):
    clf.fit(X_features[train_idx], y_class[train_idx])
    y_pred_all[test_idx] = clf.predict(X_features[test_idx])

cm = confusion_matrix(y_class, y_pred_all, labels=class_order)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

for i, c in enumerate(class_order):
    print(f"True Class: {c}")
    for j, p in enumerate(class_order):
        print(f"  -> Predicted {p}: {cm_perc[i, j]:.1f}%")
    print()
