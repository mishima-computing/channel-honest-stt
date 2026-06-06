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

def train_classifiers(speakers, extractor):
    print("Extracting clean 24kHz audio for training...")
    X_c_raw, y_c, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='consonant', cons_pre_ms=20.0
    )
    X_v_raw, y_v, _ = extractor.extract_clean_diagnostic_features(
        speakers, target_type='vowel', vowel_dur_ms=80.0
    )
    
    pipe = DegradationPipeline(target_sr=8000)
    
    print("Processing clean training features...")
    X_c_feat = []
    for x in tqdm(X_c_raw, desc="Consonants"):
        x_deg = pipe.process(x, 24000, hpf_cutoff=500.0)
        X_c_feat.append(extract_log_mel(x_deg, 8000))
        
    X_v_feat = []
    for x in tqdm(X_v_raw, desc="Vowels"):
        x_deg = pipe.process(x, 24000, hpf_cutoff=500.0)
        X_v_feat.append(extract_log_mel(x_deg, 8000))
        
    scaler_c = StandardScaler()
    scaler_v = StandardScaler()
    
    X_c_s = scaler_c.fit_transform(X_c_feat)
    X_v_s = scaler_v.fit_transform(X_v_feat)
    
    clf_c = LogisticRegression(max_iter=2000, random_state=42)
    clf_v = LogisticRegression(max_iter=2000, random_state=42)
    
    print("Training classifiers...")
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
    
    prompts = []
    answers = []
    original_texts = []
    
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
                    
        predicted_tokens = []
        original_tokens = []
        
        for start, end, ph in labels:
            if ph in ['silB', 'silE']:
                continue
            
            original_tokens.append(ph)
            
            if ph == 'pau':
                predicted_tokens.append('pau')
                continue
            if ph == 'cl':
                predicted_tokens.append('cl')
                continue
            if ph == 'N':
                predicted_tokens.append('N')
                continue
            
            is_vowel = ph in vowels
            is_cons = ph in manner_map
            
            if not is_vowel and not is_cons:
                # w, y, etc. Just keep them for now, or map them?
                # Actually w and y are approximants. In previous scripts we didn't mask them.
                predicted_tokens.append(ph)
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
                predicted_tokens.append(f"<{manner_map.get(ph, ph)}>" if is_cons else ph.lower())
                continue
                
            x_deg = pipe.process(segment, 24000, hpf_cutoff=500.0)
            feat = extract_log_mel(x_deg, 8000)
            
            if is_vowel:
                if feat.shape[0] != scaler_v.mean_.shape[0]:
                    feat = np.pad(feat, (0, max(0, scaler_v.mean_.shape[0] - feat.shape[0])))[:scaler_v.mean_.shape[0]]
                feat_s = scaler_v.transform([feat])
                pred = clf_v.predict(feat_s)[0]
                predicted_tokens.append(pred.replace('/', ''))
            else:
                if feat.shape[0] != scaler_c.mean_.shape[0]:
                    feat = np.pad(feat, (0, max(0, scaler_c.mean_.shape[0] - feat.shape[0])))[:scaler_c.mean_.shape[0]]
                feat_s = scaler_c.transform([feat])
                pred = clf_c.predict(feat_s)[0]
                predicted_tokens.append(f"<{pred}>")
                
        prompts.append(" ".join(predicted_tokens))
        answers.append(" ".join(original_tokens))
        original_texts.append(item['text'])
        
    return prompts, answers, original_texts

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    musan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/musan'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    pipe = DegradationPipeline(target_sr=8000)
    musan = MusanInjector(musan_root=musan_root)
    musan.preload(sr=24000, duration_sec=60)
    
    train_speakers = [f'jvs{i:03d}' for i in range(1, 11)] # 10 speakers for training is fast and decent
    test_speakers = [f'jvs{i:03d}' for i in range(11, 21)] # 10 different speakers for testing
    
    clf_c, scaler_c, clf_v, scaler_v = train_classifiers(train_speakers, extractor)
    
    # Generate 1000 noisy prompts at MUSAN Babble 0dB
    prompts, answers, original_texts = generate_noisy_prompts(test_speakers, 1000, 0.0, 'babble', clf_c, scaler_c, clf_v, scaler_v, extractor, musan, pipe)
    
    prompt_file = os.path.join(os.path.dirname(__file__), 'e2e_blind_prompts_0dB.txt')
    answer_file = os.path.join(os.path.dirname(__file__), 'e2e_blind_answers_0dB.txt')
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        for i, p in enumerate(prompts):
            f.write(f"[Sentence {i+1}]\nMasked: {p}\n\n")
            
    with open(answer_file, 'w', encoding='utf-8') as f:
        for i, (a, text) in enumerate(zip(answers, original_texts)):
            f.write(f"[Sentence {i+1}]\nOriginal: {text}\nPhonemes: {a}\n\n")
            
    print(f"Done. Saved to {prompt_file} and {answer_file}")

if __name__ == '__main__':
    main()
