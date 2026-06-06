import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import scipy.io.wavfile as wavfile
from tqdm import tqdm

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
    
    X_features = []
    y_labels = []
    
    target_classes = ['Flap', 'Voiced Plosive', '/a/', '/i/', '/u/']
    
    print("Extracting custom 40ms features (Flap/Plosive: [-10, 30] from label start, Vowel: [0, 40] from label start)")
    for spk in tqdm(speakers):
        lab_spk_dir = os.path.join(lab_dir, spk)
        wav_spk_dir = os.path.join(data_dir, spk, 'parallel100', 'wav24kHz16bit')
        
        if not os.path.exists(lab_spk_dir):
            continue
            
        for lab_file in os.listdir(lab_spk_dir):
            if not lab_file.endswith('.lab'):
                continue
                
            lab_path = os.path.join(lab_spk_dir, lab_file)
            wav_path = os.path.join(wav_spk_dir, lab_file.replace('.lab', '.wav'))
            
            if not os.path.exists(wav_path):
                continue
                
            phonemes = extractor.parse_lab_file(lab_path)
            
            sr, audio = wavfile.read(wav_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            # Extract samples
            for i in range(1, len(phonemes)):
                ph = phonemes[i]['phoneme']
                start = phonemes[i]['start']
                
                # Check context
                prev_ph = phonemes[i-1]['phoneme'] if i > 0 else 'sil'
                if prev_ph in ['sil', 'sp']:
                    continue
                
                label = None
                start_t = 0
                end_t = 0
                
                if ph in ['r', 'ry']:
                    label = 'Flap'
                    start_t = start - 0.010
                    end_t = start + 0.030
                elif ph in ['b', 'd', 'g', 'by', 'gy']:
                    label = 'Voiced Plosive'
                    start_t = start - 0.010
                    end_t = start + 0.030
                elif ph in ['a', 'i', 'u']:
                    label = f'/{ph}/'
                    start_t = start
                    end_t = start + 0.040
                
                if label is not None:
                    start_idx = int(start_t * sr)
                    end_idx = int(end_t * sr)
                    if start_idx >= 0 and end_idx <= len(audio):
                        x_clean = audio[start_idx:end_idx]
                        if len(x_clean) == int(0.040 * sr):
                            x_deg = pipe.process(x_clean, sr, hpf_cutoff=500.0)
                            feat = extract_log_mel(x_deg, 8000)
                            X_features.append(feat)
                            y_labels.append(label)

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
    
    print("\n--- Label-Anchored Sub-Classification ---")
    for i, c in enumerate(target_classes):
        print(f"{c}: {cm_norm[i, i]:.1%}")
        
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=target_classes)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='.1%')
    ax.set_title("Flap Diagnostic CM (Label Anchor [-10ms, +30ms])")
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), 'flap_label_diagnostic_cm.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved CM to {out_path}")

if __name__ == '__main__':
    main()
