import os
import sys
import numpy as np
import random
import librosa
from sklearn.linear_model import LogisticRegression
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

def extract_mct_training_features(speakers, musan, pipe, extractor, snr_choices=[None, 20.0, 10.0, 0.0]):
    all_sentences = []
    for spk in speakers:
        lab_spk_dir = os.path.join(extractor.lab_dir, spk)
        if not os.path.exists(lab_spk_dir):
            continue
        for f in os.listdir(lab_spk_dir):
            if f.endswith('.lab'):
                wav_path = os.path.join(extractor.data_dir, spk, 'parallel100', 'wav24kHz16bit', f.replace('.lab', '.wav'))
                if os.path.exists(wav_path):
                    all_sentences.append({'lab': os.path.join(lab_spk_dir, f), 'wav': wav_path})
                    
    manner_map = {
        'p': 'Unvoiced_Plosive', 'py': 'Unvoiced_Plosive', 't': 'Unvoiced_Plosive', 'k': 'Unvoiced_Plosive', 'ky': 'Unvoiced_Plosive',
        'ts': 'Unvoiced_Plosive', 'ch': 'Unvoiced_Plosive', 
        'b': 'Voiced_Plosive', 'by': 'Voiced_Plosive', 'd': 'Voiced_Plosive', 'g': 'Voiced_Plosive', 'gy': 'Voiced_Plosive',
        'z': 'Voiced_Plosive', 'j': 'Voiced_Plosive', 'r': 'Voiced_Plosive', 'ry': 'Voiced_Plosive',
        's': 'Unvoiced_Fricative', 'sh': 'Unvoiced_Fricative', 'h': 'Unvoiced_Fricative', 'hy': 'Unvoiced_Fricative', 'f': 'Unvoiced_Fricative',
        'm': 'Nasal', 'my': 'Nasal', 'n': 'Nasal', 'ny': 'Nasal'
    }
    vowels = ['a', 'i', 'u', 'e', 'o', 'A', 'I', 'U', 'E', 'O']
    
    X_c, y_c = [], []
    X_v, y_v = [], []
    
    np.random.seed(42)
    print(f"Extracting MCT training features from {len(all_sentences)} sentences...")
    
    for item in tqdm(all_sentences):
        audio, _ = librosa.load(item['wav'], sr=24000)
        snr = np.random.choice(snr_choices)
        
        if snr is not None:
            # We don't have random string choice easily, let's just use babble as it's the primary problem
            audio_mixed = musan.inject(audio, 24000, snr, category='babble')
        else:
            audio_mixed = audio
            
        labels = []
        with open(item['lab'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    labels.append((float(parts[0]), float(parts[1]), parts[2]))
                    
        for start, end, ph in labels:
            if ph in ['silB', 'silE', 'pau', 'cl', 'N']: continue
            is_vowel = ph in vowels
            is_cons = ph in manner_map
            if not is_vowel and not is_cons: continue
            
            if is_vowel:
                start_samp = int(start * 24000)
                end_samp = int((start + 0.08) * 24000)
            else:
                burst_time = end
                start_samp = int((burst_time - 0.02) * 24000)
                end_samp = int((burst_time + 0.04) * 24000)
                
            start_samp = max(0, start_samp)
            end_samp = min(len(audio_mixed), end_samp)
            segment = audio_mixed[start_samp:end_samp]
            if len(segment) == 0: continue
            
            x_deg = pipe.process(segment, 24000, hpf_cutoff=500.0)
            feat = extract_log_mel(x_deg, 8000)
            
            if is_vowel:
                X_v.append(feat)
                y_v.append(ph.replace('/', ''))
            else:
                X_c.append(feat)
                y_c.append(manner_map.get(ph, ph))
                
    return X_c, y_c, X_v, y_v

def train_classifiers(speakers, extractor, musan, pipe):
    print("Starting Multi-condition Training (MCT)...")
    X_c_feat, y_c, X_v_feat, y_v = extract_mct_training_features(speakers, musan, pipe, extractor)
    
    scaler_c = StandardScaler()
    scaler_v = StandardScaler()
    
    # Pad features if needed
    max_len_c = max(len(x) for x in X_c_feat)
    max_len_v = max(len(x) for x in X_v_feat)
    
    X_c_feat_pad = [np.pad(x, (0, max_len_c - len(x))) for x in X_c_feat]
    X_v_feat_pad = [np.pad(x, (0, max_len_v - len(x))) for x in X_v_feat]
    
    X_c_s = scaler_c.fit_transform(X_c_feat_pad)
    X_v_s = scaler_v.fit_transform(X_v_feat_pad)
    
    clf_c = LogisticRegression(max_iter=2000, random_state=42)
    clf_v = LogisticRegression(max_iter=2000, random_state=42)
    
    print("Training MCT classifiers...")
    clf_c.fit(X_c_s, y_c)
    clf_v.fit(X_v_s, y_v)
    
    return clf_c, scaler_c, clf_v, scaler_v

def generate_noisy_prompts(speakers, num_sentences, snr, category, clf_c, scaler_c, clf_v, scaler_v, extractor, musan, pipe):
    # Collect all available sentences for the target speakers
    all_sentences = []
    for spk in speakers:
        lab_spk_dir = os.path.join(extractor.lab_dir, spk)
        transcript_path = os.path.join(extractor.data_dir, spk, 'parallel100', 'transcripts_utf8.txt')
        
        transcripts = {}
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        transcripts[parts[0]] = parts[1]
                        
        if not os.path.exists(lab_spk_dir):
            continue
            
        for f in os.listdir(lab_spk_dir):
            if f.endswith('.lab'):
                file_id = f.replace('.lab', '')
                wav_path = os.path.join(extractor.data_dir, spk, 'parallel100', 'wav24kHz16bit', f.replace('.lab', '.wav'))
                if os.path.exists(wav_path):
                    orig_text = transcripts.get(file_id, "Unknown Transcript")
                    all_sentences.append({'spk': spk, 'id': file_id, 'lab': os.path.join(lab_spk_dir, f), 'wav': wav_path, 'text': orig_text})
    
    random.seed(42)
    selected = random.sample(all_sentences, min(num_sentences, len(all_sentences)))
    
    manner_map = {
        'p': 'Unvoiced_Plosive', 'py': 'Unvoiced_Plosive', 't': 'Unvoiced_Plosive', 'k': 'Unvoiced_Plosive', 'ky': 'Unvoiced_Plosive',
        'ts': 'Unvoiced_Plosive', 'ch': 'Unvoiced_Plosive', 
        'b': 'Voiced_Plosive', 'by': 'Voiced_Plosive', 'd': 'Voiced_Plosive', 'g': 'Voiced_Plosive', 'gy': 'Voiced_Plosive',
        'z': 'Voiced_Plosive', 'j': 'Voiced_Plosive', 'r': 'Voiced_Plosive', 'ry': 'Voiced_Plosive',
        's': 'Unvoiced_Fricative', 'sh': 'Unvoiced_Fricative', 'h': 'Unvoiced_Fricative', 'hy': 'Unvoiced_Fricative', 'f': 'Unvoiced_Fricative',
        'm': 'Nasal', 'my': 'Nasal', 'n': 'Nasal', 'ny': 'Nasal'
    }
    vowels = ['a', 'i', 'u', 'e', 'o', 'A', 'I', 'U', 'E', 'O']
    
    tokens_data = []
    
    print(f"Generating {len(selected)} noisy prompts (SNR={snr}dB, {category})...")
    for item in tqdm(selected):
        audio, _ = librosa.load(item['wav'], sr=24000)
        if snr is not None:
            audio_mixed = musan.inject(audio, 24000, snr, category=category)
        else:
            audio_mixed = audio
            
        labels = []
        with open(item['lab'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    labels.append((float(parts[0]), float(parts[1]), parts[2]))
                    
        token_idx = 0
        
        for start, end, ph in labels:
            if ph in ['silB', 'silE']:
                continue
                
            if ph == 'pau':
                continue
            if ph == 'cl':
                continue
            if ph == 'N':
                continue
            
            is_vowel = ph in vowels
            is_cons = ph in manner_map
            
            if not is_vowel and not is_cons:
                continue
                
            # Extract audio window
            if is_vowel:
                start_samp = int(start * 24000)
                end_samp = int((start + 0.08) * 24000)
            else:
                # Consonant: -20ms to +40ms relative to end (burst)
                # Wait, our JVSCVExtractor uses burst alignment. Let's approximate burst as `end` of consonant label.
                burst_time = end
                start_samp = int((burst_time - 0.02) * 24000)
                end_samp = int((burst_time + 0.04) * 24000)
                
            start_samp = max(0, start_samp)
            end_samp = min(len(audio_mixed), end_samp)
            
            segment = audio_mixed[start_samp:end_samp]
            if len(segment) == 0:
                continue
                
            x_deg = pipe.process(segment, 24000, hpf_cutoff=500.0)
            feat = extract_log_mel(x_deg, 8000)
            
            if is_vowel:
                if feat.shape[0] != scaler_v.mean_.shape[0]:
                    feat = np.pad(feat, (0, max(0, scaler_v.mean_.shape[0] - feat.shape[0])))[:scaler_v.mean_.shape[0]]
                feat_s = scaler_v.transform([feat])
                probs = clf_v.predict_proba(feat_s)[0]
                classes = clf_v.classes_
            else:
                if feat.shape[0] != scaler_c.mean_.shape[0]:
                    feat = np.pad(feat, (0, max(0, scaler_c.mean_.shape[0] - feat.shape[0])))[:scaler_c.mean_.shape[0]]
                feat_s = scaler_c.transform([feat])
                probs = clf_c.predict_proba(feat_s)[0]
                classes = clf_c.classes_
                
            sorted_indices = np.argsort(probs)[::-1]
            pred = classes[sorted_indices[0]]
            max_prob = probs[sorted_indices[0]]
            margin = max_prob - probs[sorted_indices[1]] if len(probs) > 1 else max_prob
            
            true_manner = manner_map.get(ph, ph) if is_cons else ph.replace('/', '')
            pred_manner = pred.replace('/', '')
            
            tokens_data.append({
                'sentence_id': item['id'],
                'token_idx': token_idx,
                'true_label': true_manner,
                'pred_label': pred_manner,
                'max_prob': max_prob,
                'margin': margin,
                'is_vowel': is_vowel
            })
            token_idx += 1
            
    return tokens_data

import csv

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    musan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/musan'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    pipe = DegradationPipeline(target_sr=8000)
    musan = MusanInjector(musan_root=musan_root)
    musan.preload(sr=24000, duration_sec=60)
    
    train_speakers = [f'jvs{i:03d}' for i in range(1, 11)]
    test_speakers = [f'jvs{i:03d}' for i in range(11, 21)]
    
    clf_c, scaler_c, clf_v, scaler_v = train_classifiers(train_speakers, extractor, musan, pipe)
    
    for test_snr in [10.0, 0.0]:
        tokens_data = generate_noisy_prompts(test_speakers, 1000, test_snr, 'babble', clf_c, scaler_c, clf_v, scaler_v, extractor, musan, pipe)
        
        csv_file = os.path.join(os.path.dirname(__file__), f'e2e_predictions_{int(test_snr)}dB.csv')
        
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sentence_id', 'token_idx', 'true_label', 'pred_label', 'max_prob', 'margin', 'is_vowel'])
            writer.writeheader()
            for row in tokens_data:
                writer.writerow(row)
                
        print(f"Done. Saved token predictions to {csv_file}")

if __name__ == '__main__':
    main()
