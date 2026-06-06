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
    
    target_files = []
    for lab_file in os.listdir(lab_speaker_dir):
        if not lab_file.endswith('.lab'): continue
        lab_path = os.path.join(lab_speaker_dir, lab_file)
        phonemes = extractor.parse_lab_file(lab_path)
        for i in range(1, len(phonemes)):
            if phonemes[i-1]['phoneme'] == 'r' and phonemes[i]['phoneme'] in ['a', 'i', 'u', 'e', 'o']:
                target_files.append((lab_file, i))
                if len(target_files) >= 5:
                    break
        if len(target_files) >= 5:
            break

    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    
    for row_idx, (lab_file, ph_idx) in enumerate(target_files):
        lab_path = os.path.join(lab_speaker_dir, lab_file)
        wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
        phonemes = extractor.parse_lab_file(lab_path)
        
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(wav_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        flap_start = phonemes[ph_idx-1]['start']
        flap_end = phonemes[ph_idx]['start']
        vowel_ph = phonemes[ph_idx]['phoneme']
        prev_ph = phonemes[ph_idx-2]['phoneme'] if ph_idx > 1 else 'sil'
        
        # We will extract [-20ms, +40ms] for plotting to have some margin,
        # and highlight the [-10ms, +30ms] window.
        plot_start_t = flap_start - 0.020
        plot_end_t = flap_start + 0.040
        
        start_idx = max(0, int(plot_start_t * sr))
        end_idx = int(plot_end_t * sr)
        audio_plot = audio[start_idx:end_idx]
        time_plot = np.linspace(-20, 40, len(audio_plot))
        
        ax_w = axes[row_idx, 0]
        ax_s = axes[row_idx, 1]
        
        # Waveform
        ax_w.plot(time_plot, audio_plot)
        ax_w.axvline(0, color='r', linestyle='--', label='Flap Label Start (0ms)')
        ax_w.axvline((flap_end - flap_start)*1000, color='g', linestyle='--', label='Flap Label End')
        ax_w.axvspan(-10, 30, color='y', alpha=0.3, label='Proposed [-10, 30] Window')
        ax_w.set_title(f"{prev_ph} -> r -> {vowel_ph}")
        ax_w.legend()
        
        # Spec
        S = librosa.feature.melspectrogram(y=audio_plot, sr=sr, n_fft=256, hop_length=64, n_mels=40, fmin=0, fmax=4000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, hop_length=64, x_axis='time', y_axis='mel', fmin=0, fmax=4000, ax=ax_s)
        ax_s.set_title(f"Spectrogram")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'flap_label_window_inspection.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
