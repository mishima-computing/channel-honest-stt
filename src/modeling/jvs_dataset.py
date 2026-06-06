import os
import numpy as np
import scipy.io.wavfile as wavfile
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.synthesizer.pipeline import DegradationPipeline

class JVSCVExtractor:
    def __init__(self, data_dir, lab_dir, sr_orig=24000, sr_deg=8000):
        self.data_dir = data_dir
        self.lab_dir = lab_dir
        self.sr_orig = sr_orig
        self.pipe = DegradationPipeline(sr_deg)
        
    def parse_lab_file(self, lab_path):
        """
        Parses a Julius .lab file.
        Returns a list of dicts: [{'start': 0.0, 'end': 0.1, 'phoneme': 's'}, ...]
        """
        phonemes = []
        with open(lab_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    ph = parts[2]
                    phonemes.append({'start': start, 'end': end, 'phoneme': ph})
        return phonemes

    def extract_cv_syllables_vowel_anchor(self, speakers, target_consonants, target_vowel='a', pre_vowel_ms=60.0, post_vowel_ms=40.0):
        """
        Extracts CV syllables using a fixed-length window anchored at the *vowel start*.
        This bypasses Julius's inaccurate consonant-start boundaries and naturally prevents
        zero-padding cheating in linear classifiers.
        Returns X_clean, X_deg, y, metadata.
        """
        X_clean = []
        X_deg = []
        y = []
        meta = []
        
        # Fixed length in samples
        total_duration = (pre_vowel_ms + post_vowel_ms) / 1000.0
        expected_samples = int(total_duration * self.sr_orig)
        
        for speaker in speakers:
            wav_dir = os.path.join(self.data_dir, speaker, 'parallel100', 'wav24kHz16bit')
            lab_speaker_dir = os.path.join(self.lab_dir, speaker)
            
            if not os.path.exists(wav_dir) or not os.path.exists(lab_speaker_dir):
                continue
                
            for lab_file in os.listdir(lab_speaker_dir):
                if not lab_file.endswith('.lab'):
                    continue
                    
                lab_path = os.path.join(lab_speaker_dir, lab_file)
                wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
                
                if not os.path.exists(wav_path):
                    continue
                    
                phonemes = self.parse_lab_file(lab_path)
                
                # Load audio
                sr, audio = wavfile.read(wav_path)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                
                # Find CV pairs
                for i in range(len(phonemes) - 1):
                    ph1 = phonemes[i]
                    ph2 = phonemes[i+1]
                    
                    if ph1['phoneme'] in target_consonants and ph2['phoneme'] == target_vowel:
                        # Context filter: Reject if preceded by silence/pause to avoid boundary artifacts
                        prev_ph = phonemes[i-1] if i > 0 else None
                        if prev_ph is None or prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                            continue
                            
                        cons_start = ph1['start']
                        vowel_start = ph2['start']
                        vowel_end = ph2['end']
                        
                        # Vowel-anchored slice
                        slice_start = vowel_start - (pre_vowel_ms / 1000.0)
                        slice_end = slice_start + total_duration
                        
                        start_idx = int(slice_start * self.sr_orig)
                        end_idx = start_idx + expected_samples
                        
                        if start_idx < 0 or end_idx > len(audio):
                            continue # Out of bounds
                        
                        X_clean.append(cv_audio)
                        X_deg.append(cv_deg)
                        y.append(ph1['phoneme']) # Label by consonant
                        meta.append({
                            'speaker': speaker,
                            'file': lab_file,
                            'slice_start': slice_start,
                            'cons_start_rel': cons_start - slice_start,
                            'vowel_start_rel': vowel_start - slice_start,
                            'vowel_end_rel': vowel_end - slice_start
                        })
                        
        return X_clean, X_deg, y, meta

    def extract_track_a_features(self, speakers, pre_vowel_ms=60.0, post_vowel_ms=40.0):
        """
        Extracts 12 classes (7 consonant manners + 5 vowels) using Vowel-Anchor method.
        Classes: Unvoiced Plosive, Voiced Plosive, Unvoiced Fricative, Unvoiced Affricate, 
                 Voiced Affricate/Fricative, Nasal, Flap, /a/, /i/, /u/, /e/, /o/.
        Returns: X_deg (list of waveforms), y_class (list of class names), meta.
        """
        X_deg = []
        y_class = []
        meta = []
        
        manner_map = {
            'p': 'Unvoiced Plosive', 'py': 'Unvoiced Plosive', 't': 'Unvoiced Plosive', 'k': 'Unvoiced Plosive', 'ky': 'Unvoiced Plosive',
            'b': 'Voiced Plosive', 'by': 'Voiced Plosive', 'd': 'Voiced Plosive', 'g': 'Voiced Plosive', 'gy': 'Voiced Plosive',
            's': 'Unvoiced Fricative', 'sh': 'Unvoiced Fricative', 'h': 'Unvoiced Fricative', 'hy': 'Unvoiced Fricative', 'f': 'Unvoiced Fricative',
            'ts': 'Unvoiced Affricate', 'ch': 'Unvoiced Affricate',
            'z': 'Voiced Affricate/Fricative', 'j': 'Voiced Affricate/Fricative',
            'm': 'Nasal', 'my': 'Nasal', 'n': 'Nasal', 'ny': 'Nasal',
            'r': 'Flap', 'ry': 'Flap'
        }
        vowels = ['a', 'i', 'u', 'e', 'o']
        
        total_duration = (pre_vowel_ms + post_vowel_ms) / 1000.0
        expected_samples = int(total_duration * self.sr_orig)
        
        for speaker in speakers:
            wav_dir = os.path.join(self.data_dir, speaker, 'parallel100', 'wav24kHz16bit')
            lab_speaker_dir = os.path.join(self.lab_dir, speaker)
            
            if not os.path.exists(wav_dir) or not os.path.exists(lab_speaker_dir):
                continue
                
            for lab_file in os.listdir(lab_speaker_dir):
                if not lab_file.endswith('.lab'):
                    continue
                    
                lab_path = os.path.join(lab_speaker_dir, lab_file)
                wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
                if not os.path.exists(wav_path):
                    continue
                    
                phonemes = self.parse_lab_file(lab_path)
                sr, audio = wavfile.read(wav_path)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                
                for i in range(1, len(phonemes)): # Start from 1 to safely check prev_ph
                    ph = phonemes[i]
                    prev_ph = phonemes[i-1]
                    
                    # Target 1: Consonant CV syllable
                    if ph['phoneme'] in vowels and prev_ph['phoneme'] in manner_map:
                        prev_prev_ph = phonemes[i-2] if i > 1 else None
                        # Filter out if preceded by silence
                        if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                            continue
                        
                        vowel_start = ph['start']
                        slice_start = vowel_start - (pre_vowel_ms / 1000.0)
                        start_idx = int(slice_start * self.sr_orig)
                        end_idx = start_idx + expected_samples
                        
                        if start_idx >= 0 and end_idx <= len(audio):
                            cv_audio = audio[start_idx:end_idx]
                            cv_deg = self.pipe.process(cv_audio, self.sr_orig, hpf_cutoff=500.0)
                            X_deg.append(cv_deg)
                            y_class.append(manner_map[prev_ph['phoneme']])
                            meta.append({'speaker': speaker, 'file': lab_file, 'type': 'consonant', 'phoneme': prev_ph['phoneme']})
                            
                    # Target 2: Vowel (Stable segment)
                    # We want to extract vowels that are not heavily influenced by preceding consonants if possible
                    # Or we just use any vowel but anchored. For consistency with Vowel-Anchor, 
                    # we anchor on vowel_start, but for pure vowels we might want the middle.
                    # Actually, the prompt says: "母音単体または安定母音区間". 
                    # Let's extract the Vowel-Anchored 100ms window so the shape matches exactly.
                    if ph['phoneme'] in vowels:
                        # Filter out if preceded by silence, to avoid boundary errors
                        if prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                            continue
                            
                        vowel_start = ph['start']
                        slice_start = vowel_start - (pre_vowel_ms / 1000.0)
                        start_idx = int(slice_start * self.sr_orig)
                        end_idx = start_idx + expected_samples
                        
                        if start_idx >= 0 and end_idx <= len(audio):
                            v_audio = audio[start_idx:end_idx]
                            v_deg = self.pipe.process(v_audio, self.sr_orig, hpf_cutoff=500.0)
                            X_deg.append(v_deg)
                            y_class.append(f"/{ph['phoneme']}/")
                            meta.append({'speaker': speaker, 'file': lab_file, 'type': 'vowel', 'phoneme': ph['phoneme']})

        return X_deg, y_class, meta
