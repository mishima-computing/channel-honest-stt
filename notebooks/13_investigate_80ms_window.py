import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import librosa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    speaker = 'jvs001'
    wav_dir = os.path.join(data_dir, speaker, 'parallel100', 'wav24kHz16bit')
    lab_speaker_dir = os.path.join(lab_dir, speaker)
    
    transient_manners = {
        't': 'Unvoiced Plosive',
        'd': 'Voiced Plosive',
        'ts': 'Unvoiced Affricate',
        'z': 'Voiced Affricate/Fricative'
    }
    
    samples_to_plot = {}
    
    for lab_file in os.listdir(lab_speaker_dir):
        if not lab_file.endswith('.lab'): continue
        lab_path = os.path.join(lab_speaker_dir, lab_file)
        wav_path = os.path.join(wav_dir, lab_file.replace('.lab', '.wav'))
        if not os.path.exists(wav_path): continue
        
        phonemes = extractor.parse_lab_file(lab_path)
        sr, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32) / 32768.0
        
        for i in range(1, len(phonemes)):
            ph = phonemes[i]
            prev_ph = phonemes[i-1]
            prev_prev_ph = phonemes[i-2] if i > 1 else None
            
            if ph['phoneme'] in ['a', 'i', 'u', 'e', 'o'] and prev_ph['phoneme'] in transient_manners:
                if prev_prev_ph is None or prev_prev_ph['phoneme'] in ['silB', 'sp', 'silE']:
                    continue
                
                manner = transient_manners[prev_ph['phoneme']]
                if manner not in samples_to_plot:
                    vowel_start = ph['start']
                    # Extract -120ms to +20ms around vowel anchor for visualization
                    slice_start = vowel_start - 0.120
                    slice_end = vowel_start + 0.020
                    start_idx = int(slice_start * sr)
                    end_idx = int(slice_end * sr)
                    
                    if start_idx >= 0:
                        ctx_audio = audio[start_idx:end_idx]
                        samples_to_plot[manner] = {
                            'ctx_audio': ctx_audio,
                            'ph': prev_ph['phoneme'],
                            'vowel': ph['phoneme']
                        }
                if len(samples_to_plot) == len(transient_manners):
                    break
        if len(samples_to_plot) == len(transient_manners):
            break

    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    
    for i, (manner, data) in enumerate(samples_to_plot.items()):
        ctx_audio = data['ctx_audio']
        
        # Plot waveform
        ax = axes[i, 0]
        time_axis = np.linspace(-120, 20, len(ctx_audio))
        ax.plot(time_axis, ctx_audio, color='gray', alpha=0.7)
        
        ax.axvline(x=0, color='red', linestyle='-', label='Vowel Anchor (0ms)')
        ax.axvspan(-80, 0, color='green', alpha=0.2, label='Proposed [-80ms, 0ms] Window')
        
        ax.set_title(f"{manner} (/{data['ph']}/ + /{data['vowel']}/)")
        ax.set_xlim([-120, 20])
        ax.legend(loc='upper left')
        
        # Plot Spectrogram
        ax_spec = axes[i, 1]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(ctx_audio, n_fft=512, hop_length=128)), ref=np.max)
        time_spec = np.linspace(-120, 20, D.shape[1])
        ax_spec.pcolormesh(time_spec, librosa.fft_frequencies(sr=24000, n_fft=512), D, shading='gouraud', cmap='viridis')
        
        ax_spec.axvline(x=0, color='red', linestyle='-', label='Vowel Anchor')
        ax_spec.axvspan(-80, 0, color='green', alpha=0.2, label='Proposed Window')
        
        ax_spec.set_title(f"Spectrogram")
        ax_spec.set_xlim([-120, 20])
        ax_spec.set_ylim([0, 8000])

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'vowel_anchor_80ms_test.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()
