import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    
    speaker = 'jvs001'
    wav_dir = os.path.join(data_dir, speaker, 'parallel100', 'wav24kHz16bit')
    lab_speaker_dir = os.path.join(lab_dir, speaker)
    
    # Find a few lab files with 'r'
    target_files = []
    for lab_file in os.listdir(lab_speaker_dir):
        if not lab_file.endswith('.lab'): continue
        lab_path = os.path.join(lab_speaker_dir, lab_file)
        phonemes = extractor.parse_lab_file(lab_path)
        for i in range(1, len(phonemes)):
            if phonemes[i-1]['phoneme'] == 'r' and phonemes[i]['phoneme'] in ['a', 'i', 'u', 'e', 'o']:
                # Found a Flap -> Vowel
                target_files.append((lab_file, i))
                if len(target_files) >= 3:
                    break
        if len(target_files) >= 3:
            break

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    # Rows: 3 different samples
    # Cols: Waveform(-80ms), Spec(-80ms), Waveform(-30ms), Spec(-30ms)
    
    for row_idx, (lab_file, ph_idx) in enumerate(target_files):
        lab_path = os.path.join(lab_speaker_dir, lab_file)
        wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
        phonemes = extractor.parse_lab_file(lab_path)
        
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(wav_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        vowel_start = phonemes[ph_idx]['start']
        vowel_ph = phonemes[ph_idx]['phoneme']
        flap_start = phonemes[ph_idx-1]['start']
        prev_ph = phonemes[ph_idx-2]['phoneme'] if ph_idx > 1 else 'sil'
        
        # Extractor logic:
        # -80ms window: [vowel_start - 0.080, vowel_start]
        # -30ms window: [vowel_start - 0.030, vowel_start]
        
        # Plot -80ms window
        start_80 = max(0, int((vowel_start - 0.080) * sr))
        end_80 = int(vowel_start * sr)
        audio_80 = audio[start_80:end_80]
        
        # Plot -30ms window
        start_30 = max(0, int((vowel_start - 0.030) * sr))
        end_30 = int(vowel_start * sr)
        audio_30 = audio[start_30:end_30]
        
        time_80 = np.linspace(-80, 0, len(audio_80))
        time_30 = np.linspace(-30, 0, len(audio_30))
        
        # Mark actual flap boundaries
        # Flap start is at (flap_start - vowel_start) * 1000
        flap_start_ms = (flap_start - vowel_start) * 1000
        
        ax_w80 = axes[row_idx, 0]
        ax_s80 = axes[row_idx, 1]
        ax_w30 = axes[row_idx, 2]
        ax_s30 = axes[row_idx, 3]
        
        # Waveform 80ms
        ax_w80.plot(time_80, audio_80)
        ax_w80.axvline(flap_start_ms, color='r', linestyle='--', label=f'Flap start ({flap_start_ms:.1f}ms)')
        ax_w80.set_title(f"[-80ms] {prev_ph} -> r -> {vowel_ph}")
        ax_w80.legend()
        
        # Spec 80ms
        S_80 = librosa.feature.melspectrogram(y=audio_80, sr=sr, n_fft=256, hop_length=64, n_mels=40, fmin=0, fmax=4000)
        S_dB_80 = librosa.power_to_db(S_80, ref=np.max)
        librosa.display.specshow(S_dB_80, sr=sr, hop_length=64, x_axis='time', y_axis='mel', fmin=0, fmax=4000, ax=ax_s80)
        ax_s80.set_title(f"[-80ms] Spectrogram")
        
        # Waveform 30ms
        ax_w30.plot(time_30, audio_30)
        ax_w30.axvline(flap_start_ms, color='r', linestyle='--')
        ax_w30.set_title(f"[-30ms] {prev_ph} -> r -> {vowel_ph}")
        
        # Spec 30ms
        S_30 = librosa.feature.melspectrogram(y=audio_30, sr=sr, n_fft=256, hop_length=64, n_mels=40, fmin=0, fmax=4000)
        S_dB_30 = librosa.power_to_db(S_30, ref=np.max)
        librosa.display.specshow(S_dB_30, sr=sr, hop_length=64, x_axis='time', y_axis='mel', fmin=0, fmax=4000, ax=ax_s30)
        ax_s30.set_title(f"[-30ms] Spectrogram")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'flap_window_inspection.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
