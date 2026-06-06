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

def extract_zj_split_80ms(extractor, speakers):
    X_deg = []
    y_class = []
    
    # Target classes for this diagnostic
    target_classes = {
        'b': 'Voiced Plosive', 'd': 'Voiced Plosive', 'g': 'Voiced Plosive',
        'p': 'Unvoiced Plosive', 't': 'Unvoiced Plosive', 'k': 'Unvoiced Plosive',
        's': 'Unvoiced Fricative', 'sh': 'Unvoiced Fricative', 'f': 'Unvoiced Fricative', 'h': 'Unvoiced Fricative',
    }
    
    vowels = ['a', 'i', 'u', 'e', 'o']
    
    for speaker in speakers:
        wav_dir = os.path.join(extractor.data_dir, speaker, 'parallel100', 'wav24kHz16bit')
        lab_speaker_dir = os.path.join(extractor.lab_dir, speaker)
        if not os.path.exists(wav_dir) or not os.path.exists(lab_speaker_dir): continue
            
        for lab_file in os.listdir(lab_speaker_dir):
            if not lab_file.endswith('.lab'): continue
            lab_path = os.path.join(lab_speaker_dir, lab_file)
            wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
            if not os.path.exists(wav_path): continue
                
            phonemes = extractor.parse_lab_file(lab_path)
            import scipy.io.wavfile as wavfile
            sr, audio = wavfile.read(wav_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            for i in range(1, len(phonemes)):
                ph = phonemes[i]
                prev_ph = phonemes[i-1]
                prev_prev_ph = phonemes[i-2] if i > 1 else None
                
                if ph['phoneme'] in vowels:
                    c_class = None
                    if prev_ph['phoneme'] in ['z', 'j']:
                        # Determine context for z/j
                        if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE', 'N']:
                            c_class = 'Voiced Affricate [dz]' # Initial
                        elif prev_prev_ph['phoneme'] in vowels:
                            c_class = 'Voiced Fricative [z]' # Medial
                    elif prev_ph['phoneme'] in target_classes:
                        # For other classes, we usually skip post-silence to avoid alignment errors, 
                        # but since we use vowel anchor, we can include them, OR we can just include medial to match.
                        # Let's include all to get enough samples, but skip post-silence to be safe for normal classes
                        if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                            continue
                        c_class = target_classes[prev_ph['phoneme']]
                        
                    if c_class:
                        vowel_start = ph['start']
                        slice_start = vowel_start - 0.080
                        slice_end = vowel_start
                        
                        start_idx = int(slice_start * extractor.sr_orig)
                        end_idx = int(slice_end * extractor.sr_orig)
                        
                        if start_idx >= 0 and end_idx <= len(audio):
                            c_audio = audio[start_idx:end_idx]
                            c_deg = extractor.pipe.process(c_audio, extractor.sr_orig, hpf_cutoff=500.0)
                            X_deg.append(c_deg)
                            y_class.append(c_class)

    return X_deg, y_class


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    
    print("Extracting z/j diagnostic features with Vowel-Anchor [-80ms, 0ms] window...")
    X_deg, y_class = extract_zj_split_80ms(extractor, speakers)
    
    feat_extractor = FeatureExtractor(sr=8000)
    X_features = [feat_extractor.extract_log_mel(sig).flatten() for sig in X_deg]
    X_features = np.array(X_features)
    y_class = np.array(y_class)
    
    print(f"Feature matrix shape: {X_features.shape}")
    from collections import Counter
    print(f"Class distribution: {Counter(y_class)}")
    
    class_order = [
        'Unvoiced Plosive', 'Voiced Plosive', 
        'Unvoiced Fricative',
        'Voiced Affricate [dz]', 'Voiced Fricative [z]'
    ]
    
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_all = np.zeros_like(y_class)
    
    print("Training Logistic Regression...")
    for train_idx, test_idx in skf.split(X_features, y_class):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y_class[train_idx], y_class[test_idx]
        clf.fit(X_train, y_train)
        y_pred_all[test_idx] = clf.predict(X_test)
        
    print("\nClassification Report:")
    print(classification_report(y_class, y_pred_all, labels=class_order))
    
    cm = confusion_matrix(y_class, y_pred_all, labels=class_order)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_order, yticklabels=class_order)
    plt.title("Diagnostic: z/j Split (Initial vs Medial context)")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    
    out_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix_zj_split.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
