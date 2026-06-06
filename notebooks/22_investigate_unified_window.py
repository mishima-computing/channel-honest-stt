import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wavfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor

def detect_burst(audio, sr):
    """Detect burst using pre-emphasis and RMS derivative."""
    frame_length = int(0.005 * sr)
    hop_length = int(0.001 * sr)
    audio_hp = librosa.effects.preemphasis(audio)
    rms = librosa.feature.rms(y=audio_hp, frame_length=frame_length, hop_length=hop_length)[0]
    rms_diff = np.diff(rms, prepend=rms[0])
    peak_frame = np.argmax(rms_diff)
    return peak_frame * hop_length

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    
    speaker = 'jvs001'
    wav_dir = os.path.join(data_dir, speaker, 'parallel100', 'wav24kHz16bit')
    lab_speaker_dir = os.path.join(lab_dir, speaker)
    
    # We want one example of each of the 5 main categories in the 9-class system
    categories = {
        'Unvoiced Plosive (k)': {'target': 'k', 'anchor_type': 'burst'},
        'Voiced Plosive (d)': {'target': 'd', 'anchor_type': 'burst'},
        'Voiced Plosive/Flap (r)': {'target': 'r', 'anchor_type': 'label'},
        'Unvoiced Fricative (s)': {'target': 's', 'anchor_type': 'label'},
        'Nasal (m)': {'target': 'm', 'anchor_type': 'label'},
        'Vowel (a)': {'target': 'a', 'anchor_type': 'label'}
    }
    
    examples = {k: None for k in categories.keys()}
    
    for lab_file in os.listdir(lab_speaker_dir):
        if not lab_file.endswith('.lab'): continue
        lab_path = os.path.join(lab_speaker_dir, lab_file)
        phonemes = extractor.parse_lab_file(lab_path)
        
        for i in range(1, len(phonemes)):
            ph = phonemes[i]['phoneme']
            for cat_name, cat_info in categories.items():
                if examples[cat_name] is None and ph == cat_info['target']:
                    # Need enough context
                    if phonemes[i]['start'] > 0.2:
                        examples[cat_name] = (lab_file, i)
                        
        if all(v is not None for v in examples.values()):
            break

    fig, axes = plt.subplots(len(categories), 2, figsize=(12, 3 * len(categories)))
    
    for row_idx, (cat_name, (lab_file, ph_idx)) in enumerate(examples.items()):
        lab_path = os.path.join(lab_speaker_dir, lab_file)
        wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
        phonemes = extractor.parse_lab_file(lab_path)
        
        sr, audio = wavfile.read(wav_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        ph_info = phonemes[ph_idx]
        ph_start = ph_info['start']
        anchor_type = categories[cat_name]['anchor_type']
        
        if anchor_type == 'burst':
            # Search for burst in [-150ms, 0] relative to following vowel
            # Wait, if ph is a consonant, next is usually vowel
            next_start = phonemes[ph_idx+1]['start'] if ph_idx+1 < len(phonemes) else ph_start + 0.1
            search_start_idx = max(0, int((ph_start - 0.050) * sr))
            search_end_idx = int(next_start * sr)
            search_audio = audio[search_start_idx:search_end_idx]
            burst_offset = detect_burst(search_audio, sr)
            anchor_idx = search_start_idx + burst_offset
            anchor_t = anchor_idx / sr
        else:
            anchor_t = ph_start
            anchor_idx = int(anchor_t * sr)
            
        # Extract [-20ms, +40ms] around anchor
        plot_start_t = anchor_t - 0.020
        plot_end_t = anchor_t + 0.040
        
        start_idx = max(0, int(plot_start_t * sr))
        end_idx = int(plot_end_t * sr)
        audio_plot = audio[start_idx:end_idx]
        time_plot = np.linspace(-20, 40, len(audio_plot))
        
        ax_w = axes[row_idx, 0]
        ax_s = axes[row_idx, 1]
        
        # Waveform
        ax_w.plot(time_plot, audio_plot)
        ax_w.axvline(0, color='r', linestyle='--', label='Anchor (0ms)')
        
        # Draw label boundaries relative to anchor
        rel_start = (ph_start - anchor_t) * 1000
        rel_end = (phonemes[ph_idx]['end'] - anchor_t) * 1000
        ax_w.axvspan(rel_start, rel_end, color='y', alpha=0.2, label='Label region')
        
        ax_w.set_title(f"{cat_name} (Anchor: {anchor_type})")
        if row_idx == 0:
            ax_w.legend()
        
        # Spec
        S = librosa.feature.melspectrogram(y=audio_plot, sr=sr, n_fft=256, hop_length=64, n_mels=40, fmin=0, fmax=4000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, hop_length=64, x_axis='time', y_axis='mel', fmin=0, fmax=4000, ax=ax_s)
        ax_s.set_title(f"Spectrogram")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'unified_window_inspection.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
