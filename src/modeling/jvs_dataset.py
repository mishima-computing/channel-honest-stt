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

    def extract_cv_syllables(self, speakers, target_consonants, target_vowel='a', pre_margin_ms=30.0, post_margin_ms=10.0):
        """
        Scans the lab files for the specified speakers, extracts CV syllables
        matching (target_consonant + target_vowel), and applies Phase 1 degradation.
        Includes a time margin (pre/post) to absorb alignment quantization errors.
        Returns X_clean, X_deg, y, metadata.
        """
        X_clean = []
        X_deg = []
        y = []
        meta = []
        
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
                        start_time = ph1['start']
                        vowel_start = ph2['start']
                        end_time = ph2['end']
                        
                        # Apply Margins
                        slice_start = max(0.0, start_time - (pre_margin_ms / 1000.0))
                        slice_end = min(len(audio) / self.sr_orig, end_time + (post_margin_ms / 1000.0))
                        
                        start_idx = int(slice_start * self.sr_orig)
                        end_idx = int(slice_end * self.sr_orig)
                        
                        # Extract clean CV syllable
                        cv_audio = audio[start_idx:end_idx]
                        
                        # Apply Phase 1 degradation
                        cv_deg = self.pipe.process(cv_audio, self.sr_orig, hpf_cutoff=500.0)
                        
                        X_clean.append(cv_audio)
                        X_deg.append(cv_deg)
                        y.append(ph1['phoneme']) # Label by consonant
                        meta.append({
                            'speaker': speaker,
                            'file': lab_file,
                            'slice_start': slice_start,
                            'cons_start_rel': start_time - slice_start,
                            'vowel_start_rel': vowel_start - slice_start,
                            'vowel_end_rel': end_time - slice_start
                        })
                        
        return X_clean, X_deg, y, meta
