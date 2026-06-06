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
            'r': 'Voiced Plosive', 'ry': 'Voiced Plosive'
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

    def extract_diagnostic_features(self, speakers, target_type='vowel', cons_pre_ms=60.0, vowel_dur_ms=60.0):
        """
        Diagnostic extraction separating consonants and vowels to prevent absorption artifacts.
        - target_type='vowel': Extracts 5 vowels. Window = [0ms, +vowel_dur_ms] relative to vowel anchor.
        - target_type='consonant': Extracts 7 consonant manners. Window = [-cons_pre_ms, 0ms] relative to vowel anchor.
        Returns: X_deg, y_class, meta
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
            'r': 'Voiced Plosive', 'ry': 'Voiced Plosive'
        }
        vowels = ['a', 'i', 'u', 'e', 'o']
        
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
                
                for i in range(1, len(phonemes)):
                    ph = phonemes[i]
                    prev_ph = phonemes[i-1]
                    prev_prev_ph = phonemes[i-2] if i > 1 else None
                    
                    if target_type == 'consonant':
                        if ph['phoneme'] in vowels and prev_ph['phoneme'] in manner_map:
                            if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                                continue
                            
                            vowel_start = ph['start']
                            slice_start = vowel_start - (cons_pre_ms / 1000.0)
                            slice_end = vowel_start # Exclude the vowel completely
                            
                            start_idx = int(slice_start * self.sr_orig)
                            end_idx = int(slice_end * self.sr_orig)
                            
                            if start_idx >= 0 and end_idx <= len(audio):
                                c_audio = audio[start_idx:end_idx]
                                c_deg = self.pipe.process(c_audio, self.sr_orig, hpf_cutoff=500.0)
                                X_deg.append(c_deg)
                                y_class.append(manner_map[prev_ph['phoneme']])
                                meta.append({'speaker': speaker, 'file': lab_file})
                                
                    elif target_type == 'vowel':
                        if ph['phoneme'] in vowels:
                            if prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                                continue
                                
                            vowel_start = ph['start']
                            slice_start = vowel_start
                            slice_end = vowel_start + (vowel_dur_ms / 1000.0)
                            
                            start_idx = int(slice_start * self.sr_orig)
                            end_idx = int(slice_end * self.sr_orig)
                            
                            if start_idx >= 0 and end_idx <= len(audio):
                                v_audio = audio[start_idx:end_idx]
                                v_deg = self.pipe.process(v_audio, self.sr_orig, hpf_cutoff=500.0)
                                X_deg.append(v_deg)
                                y_class.append(f"/{ph['phoneme']}/")
                                meta.append({'speaker': speaker, 'file': lab_file})

        return X_deg, y_class, meta

    def extract_clean_diagnostic_features(self, speakers, target_type='vowel', cons_pre_ms=80.0, vowel_dur_ms=80.0):
        """
        Extracts CLEAN 24kHz audio chunks for Level 2 noise injection tests.
        Returns: X_clean_24k, y_class, meta
        """
        X_clean = []
        y_class = []
        meta = []
        
        manner_map = {
            'p': 'Unvoiced Plosive', 'py': 'Unvoiced Plosive', 't': 'Unvoiced Plosive', 'k': 'Unvoiced Plosive', 'ky': 'Unvoiced Plosive',
            'ts': 'Unvoiced Plosive', 'ch': 'Unvoiced Plosive', # Merged
            'b': 'Voiced Plosive', 'by': 'Voiced Plosive', 'd': 'Voiced Plosive', 'g': 'Voiced Plosive', 'gy': 'Voiced Plosive',
            'z': 'Voiced Plosive', 'j': 'Voiced Plosive', # Merged
            's': 'Unvoiced Fricative', 'sh': 'Unvoiced Fricative', 'h': 'Unvoiced Fricative', 'hy': 'Unvoiced Fricative', 'f': 'Unvoiced Fricative',
            'm': 'Nasal', 'my': 'Nasal', 'n': 'Nasal', 'ny': 'Nasal',
            'r': 'Voiced Plosive', 'ry': 'Voiced Plosive'
        }
        vowels = ['a', 'i', 'u', 'e', 'o']
        
        for speaker in speakers:
            wav_dir = os.path.join(self.data_dir, speaker, 'parallel100', 'wav24kHz16bit')
            lab_speaker_dir = os.path.join(self.lab_dir, speaker)
            
            if not os.path.exists(wav_dir) or not os.path.exists(lab_speaker_dir): continue
                
            for lab_file in os.listdir(lab_speaker_dir):
                if not lab_file.endswith('.lab'): continue
                lab_path = os.path.join(lab_speaker_dir, lab_file)
                wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
                if not os.path.exists(wav_path): continue
                    
                phonemes = self.parse_lab_file(lab_path)
                import scipy.io.wavfile as wavfile
                sr, audio = wavfile.read(wav_path)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                
                for i in range(1, len(phonemes)):
                    ph = phonemes[i]
                    prev_ph = phonemes[i-1]
                    prev_prev_ph = phonemes[i-2] if i > 1 else None
                    
                    if target_type == 'consonant':
                        if ph['phoneme'] in vowels and prev_ph['phoneme'] in manner_map:
                            if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                                continue
                            
                            vowel_start = ph['start']
                            slice_start = vowel_start - (cons_pre_ms / 1000.0)
                            slice_end = vowel_start
                            
                            start_idx = int(slice_start * self.sr_orig)
                            expected_samples = int((cons_pre_ms / 1000.0) * self.sr_orig)
                            end_idx = start_idx + expected_samples
                            
                            if start_idx >= 0 and end_idx <= len(audio):
                                c_audio = audio[start_idx:end_idx]
                                if len(c_audio) == expected_samples:
                                    X_clean.append(c_audio)
                                    y_class.append(manner_map[prev_ph['phoneme']])
                                    meta.append({'speaker': speaker, 'file': lab_file, 'phoneme': prev_ph['phoneme']})
                                
                    elif target_type == 'vowel':
                        if ph['phoneme'] in vowels:
                            if prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                                continue
                                
                            vowel_start = ph['start']
                            slice_start = vowel_start
                            
                            start_idx = int(slice_start * self.sr_orig)
                            expected_samples = int((vowel_dur_ms / 1000.0) * self.sr_orig)
                            end_idx = start_idx + expected_samples
                            
                            if start_idx >= 0 and end_idx <= len(audio):
                                v_audio = audio[start_idx:end_idx]
                                if len(v_audio) == expected_samples:
                                    X_clean.append(v_audio)
                                    y_class.append(f"/{ph['phoneme']}/")
                                    meta.append({'speaker': speaker, 'file': lab_file, 'phoneme': ph['phoneme']})

        return X_clean, y_class, meta

    def extract_transient_burst_aligned(self, speakers, target_type='transient', burst_pre_ms=20.0, burst_post_ms=40.0):
        """
        Diagnostic extraction specifically for transient sounds (plosives, affricates),
        aligned to the detected burst rather than a fixed vowel anchor.
        Window = [-burst_pre_ms, +burst_post_ms] relative to the detected burst.
        """
        import librosa
        X_deg = []
        y_class = []
        meta = []
        
        manner_map = {
            'p': 'Unvoiced Plosive', 'py': 'Unvoiced Plosive', 't': 'Unvoiced Plosive', 'k': 'Unvoiced Plosive', 'ky': 'Unvoiced Plosive',
            'b': 'Voiced Plosive', 'by': 'Voiced Plosive', 'd': 'Voiced Plosive', 'g': 'Voiced Plosive', 'gy': 'Voiced Plosive',
            'ts': 'Unvoiced Affricate', 'ch': 'Unvoiced Affricate',
            'z': 'Voiced Affricate/Fricative', 'j': 'Voiced Affricate/Fricative',
        }
        vowels = ['a', 'i', 'u', 'e', 'o']
        
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
                
                for i in range(1, len(phonemes)):
                    ph = phonemes[i]
                    prev_ph = phonemes[i-1]
                    prev_prev_ph = phonemes[i-2] if i > 1 else None
                    
                    if ph['phoneme'] in vowels and prev_ph['phoneme'] in manner_map:
                        if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                            continue
                            
                        vowel_start = ph['start']
                        search_start = vowel_start - 0.150
                        if search_start < 0:
                            continue
                            
                        # Detect burst
                        start_idx = int(search_start * self.sr_orig)
                        end_idx = int(vowel_start * self.sr_orig)
                        c_audio = audio[start_idx:end_idx]
                        
                        frame_length = int(0.005 * self.sr_orig)
                        hop_length = int(0.001 * self.sr_orig)
                        
                        # Use preemphasis to suppress low frequencies and highlight bursts/friction
                        audio_hp = librosa.effects.preemphasis(c_audio)
                        rms = librosa.feature.rms(y=audio_hp, frame_length=frame_length, hop_length=hop_length)[0]
                        rms_diff = np.diff(rms, prepend=rms[0])
                        peak_frame = np.argmax(rms_diff)
                        burst_sample_offset = peak_frame * hop_length
                        burst_idx = start_idx + burst_sample_offset
                        
                        # Extract around burst
                        slice_start_idx = burst_idx - int((burst_pre_ms / 1000.0) * self.sr_orig)
                        slice_end_idx = burst_idx + int((burst_post_ms / 1000.0) * self.sr_orig)
                        
                        if slice_start_idx >= 0 and slice_end_idx <= len(audio):
                            transient_audio = audio[slice_start_idx:slice_end_idx]
                            c_deg = self.pipe.process(transient_audio, self.sr_orig, hpf_cutoff=500.0)
                            X_deg.append(c_deg)
                            y_class.append(manner_map[prev_ph['phoneme']])
                            meta.append({'speaker': speaker, 'file': lab_file})

        return X_deg, y_class, meta

    def extract_transient_vowel_anchor_80ms(self, speakers):
        """
        Extracts a fixed 80ms window [-80ms, 0ms] relative to the vowel anchor
        specifically for the 4 transient classes to capture full friction without vowel absorption.
        """
        import librosa
        X_deg = []
        y_class = []
        meta = []
        
        manner_map = {
            'p': 'Unvoiced Plosive', 'py': 'Unvoiced Plosive', 't': 'Unvoiced Plosive', 'k': 'Unvoiced Plosive', 'ky': 'Unvoiced Plosive',
            'b': 'Voiced Plosive', 'by': 'Voiced Plosive', 'd': 'Voiced Plosive', 'g': 'Voiced Plosive', 'gy': 'Voiced Plosive',
            'ts': 'Unvoiced Affricate', 'ch': 'Unvoiced Affricate',
            'z': 'Voiced Affricate/Fricative', 'j': 'Voiced Affricate/Fricative',
        }
        vowels = ['a', 'i', 'u', 'e', 'o']
        
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
                
                for i in range(1, len(phonemes)):
                    ph = phonemes[i]
                    prev_ph = phonemes[i-1]
                    prev_prev_ph = phonemes[i-2] if i > 1 else None
                    
                    if ph['phoneme'] in vowels and prev_ph['phoneme'] in manner_map:
                        if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                            continue
                            
                        vowel_start = ph['start']
                        # Extract [-80ms, 0ms]
                        slice_start = vowel_start - 0.080
                        slice_end = vowel_start
                        
                        start_idx = int(slice_start * self.sr_orig)
                        end_idx = int(slice_end * self.sr_orig)
                        
                        if start_idx >= 0 and end_idx <= len(audio):
                            c_audio = audio[start_idx:end_idx]
                            c_deg = self.pipe.process(c_audio, self.sr_orig, hpf_cutoff=500.0)
                            X_deg.append(c_deg)
                            y_class.append(manner_map[prev_ph['phoneme']])
                            meta.append({'speaker': speaker, 'file': lab_file})

        return X_deg, y_class, meta

    def extract_dual_window_features(self, speakers, target_type='all'):
        """
        Extracts features using a Dual Window architecture (60ms fixed dimension for all):
        - Consonants: [-20ms, +40ms] anchored to burst or label.
        - Vowels: [0ms, +60ms] anchored to vowel label.
        Returns X_deg, y_class, meta.
        """
        import librosa
        X_deg = []
        y_class = []
        meta = []
        
        manner_map = {
            'p': 'Unvoiced Plosive', 'py': 'Unvoiced Plosive', 't': 'Unvoiced Plosive', 'k': 'Unvoiced Plosive', 'ky': 'Unvoiced Plosive',
            'ts': 'Unvoiced Plosive', 'ch': 'Unvoiced Plosive',
            'b': 'Voiced Plosive', 'by': 'Voiced Plosive', 'd': 'Voiced Plosive', 'g': 'Voiced Plosive', 'gy': 'Voiced Plosive',
            'z': 'Voiced Plosive', 'j': 'Voiced Plosive',
            'r': 'Voiced Plosive', 'ry': 'Voiced Plosive',
            's': 'Unvoiced Fricative', 'sh': 'Unvoiced Fricative', 'h': 'Unvoiced Fricative', 'hy': 'Unvoiced Fricative', 'f': 'Unvoiced Fricative',
            'm': 'Nasal', 'my': 'Nasal', 'n': 'Nasal', 'ny': 'Nasal', 'N': 'Nasal'
        }
        vowels = ['a', 'i', 'u', 'e', 'o']
        
        burst_anchors = ['p', 'py', 't', 'k', 'ky', 'b', 'by', 'd', 'g', 'gy', 'ts', 'ch']
        
        for speaker in speakers:
            wav_dir = os.path.join(self.data_dir, speaker, 'parallel100', 'wav24kHz16bit')
            lab_speaker_dir = os.path.join(self.lab_dir, speaker)
            
            if not os.path.exists(wav_dir) or not os.path.exists(lab_speaker_dir): continue
                
            for lab_file in os.listdir(lab_speaker_dir):
                if not lab_file.endswith('.lab'): continue
                lab_path = os.path.join(lab_speaker_dir, lab_file)
                wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
                if not os.path.exists(wav_path): continue
                    
                phonemes = self.parse_lab_file(lab_path)
                sr, audio = wavfile.read(wav_path)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                
                for i in range(1, len(phonemes)):
                    ph = phonemes[i]
                    ph_name = ph['phoneme']
                    prev_ph_name = phonemes[i-1]['phoneme'] if i > 0 else None
                    
                    is_valid_consonant = ph_name in vowels and prev_ph_name in manner_map
                    is_valid_vowel = ph_name in vowels and prev_ph_name not in ['silB', 'sp', 'silE']
                    
                    extract_ph = None
                    extract_class = None
                    extract_start = None
                    next_start = None
                    is_vowel_target = False
                    
                    if (target_type in ['all', 'consonant']) and is_valid_consonant:
                        extract_ph = prev_ph_name
                        extract_class = manner_map[prev_ph_name]
                        extract_start = phonemes[i-1]['start']
                        next_start = ph['start']
                        is_vowel_target = False
                    elif (target_type in ['all', 'vowel']) and is_valid_vowel:
                        extract_ph = ph_name
                        extract_class = f"/{ph_name}/"
                        extract_start = ph['start']
                        next_start = phonemes[i+1]['start'] if i+1 < len(phonemes) else ph['end']
                        is_vowel_target = True
                        
                    if extract_ph is None:
                        continue
                        
                    if is_vowel_target:
                        anchor_t = extract_start
                        slice_start = anchor_t
                        slice_end = anchor_t + 0.060
                    else:
                        if extract_ph in burst_anchors:
                            search_start = max(0, extract_start - 0.050)
                            search_end = next_start
                            start_idx = int(search_start * self.sr_orig)
                            end_idx = int(search_end * self.sr_orig)
                            search_audio = audio[start_idx:end_idx]
                            if len(search_audio) == 0: continue
                                
                            audio_hp = librosa.effects.preemphasis(search_audio)
                            frame_length = int(0.005 * self.sr_orig)
                            hop_length = int(0.001 * self.sr_orig)
                            rms = librosa.feature.rms(y=audio_hp, frame_length=frame_length, hop_length=hop_length)[0]
                            rms_diff = np.diff(rms, prepend=rms[0])
                            burst_offset = np.argmax(rms_diff) * hop_length
                            anchor_t = search_start + (burst_offset / self.sr_orig)
                        else:
                            anchor_t = extract_start
                            
                        slice_start = anchor_t - 0.020
                        slice_end = anchor_t + 0.040
                    
                    start_idx = int(slice_start * self.sr_orig)
                    end_idx = int(slice_end * self.sr_orig)
                    
                    if start_idx >= 0 and end_idx <= len(audio):
                        c_audio = audio[start_idx:end_idx]
                        if len(c_audio) == int(0.060 * self.sr_orig):
                            c_deg = self.pipe.process(c_audio, self.sr_orig, hpf_cutoff=500.0)
                            X_deg.append(c_deg)
                            y_class.append(extract_class)
                            meta.append({'speaker': speaker, 'file': lab_file, 'phoneme': extract_ph, 'anchor_t': anchor_t})

        return X_deg, y_class, meta
