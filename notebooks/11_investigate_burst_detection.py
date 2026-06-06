import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import librosa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor

def detect_burst(audio, sr):
    """
    Detects the burst in a short audio segment (e.g. -150ms to 0ms before vowel).
    We look for the maximum derivative of energy (the sharpest onset).
    """
    # Calculate short-time RMS energy (small window for high temporal resolution)
    frame_length = int(0.005 * sr) # 5ms
    hop_length = int(0.001 * sr) # 1ms
    
    # Optional: high-pass filter to focus on burst/friction, removing low freq murmur
    audio_hp = librosa.effects.preemphasis(audio)
    
    rms = librosa.feature.rms(y=audio_hp, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calculate derivative of RMS
    rms_diff = np.diff(rms, prepend=rms[0])
    
    # Find the peak of the derivative
    peak_frame = np.argmax(rms_diff)
    peak_sample = peak_frame * hop_length
    
    return peak_sample, rms, rms_diff, hop_length

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
        'z': 'Voiced Affricate/Fricative' # Note: 'z' is often realized as an affricate [dz] at word starts in Japanese
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
            
            if ph['phoneme'] in ['a', 'i', 'u', 'e', 'o'] and prev_ph['phoneme'] in transient_manners:
                manner = transient_manners[prev_ph['phoneme']]
                if manner not in samples_to_plot:
                    # Extract -150ms to 0ms (vowel anchor)
                    vowel_start = ph['start']
                    slice_start = vowel_start - 0.150
                    start_idx = int(slice_start * sr)
                    end_idx = int(vowel_start * sr)
                    
                    if start_idx >= 0:
                        c_audio = audio[start_idx:end_idx]
                        
                        # also get some context for spectrogram (-150ms to +50ms)
                        ctx_end = int((vowel_start + 0.050) * sr)
                        ctx_audio = audio[start_idx:ctx_end]
                        
                        burst_idx, rms, rms_diff, hop = detect_burst(c_audio, sr)
                        
                        samples_to_plot[manner] = {
                            'audio': c_audio,
                            'ctx_audio': ctx_audio,
                            'burst_idx': burst_idx,
                            'rms': rms,
                            'rms_diff': rms_diff,
                            'hop': hop,
                            'ph': prev_ph['phoneme'],
                            'vowel': ph['phoneme']
                        }
                if len(samples_to_plot) == len(transient_manners):
                    break
        if len(samples_to_plot) == len(transient_manners):
            break

    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    
    for i, (manner, data) in enumerate(samples_to_plot.items()):
        c_audio = data['audio']
        ctx_audio = data['ctx_audio']
        burst_idx = data['burst_idx']
        
        # Plot waveform
        ax = axes[i, 0]
        time_axis = np.linspace(-150, 0, len(c_audio))
        ax.plot(time_axis, c_audio, color='gray', alpha=0.7, label='Waveform')
        
        # Plot RMS diff (scaled)
        time_rms = np.linspace(-150, 0, len(data['rms_diff']))
        ax.plot(time_rms, data['rms_diff'] / np.max(np.abs(data['rms_diff'])) * np.max(np.abs(c_audio)), 
                color='red', label='Energy Diff')
        
        # Mark burst
        burst_time_ms = -150 + (burst_idx / 24000.0 * 1000)
        ax.axvline(x=burst_time_ms, color='blue', linestyle='--', label='Detected Burst')
        
        # Mark new extraction window (e.g. -20ms to +40ms around burst)
        ax.axvspan(burst_time_ms - 20, burst_time_ms + 40, color='green', alpha=0.2, label='New Window')
        
        ax.set_title(f"{manner} (/{data['ph']}/ + /{data['vowel']}/)")
        ax.set_xlim([-150, 0])
        ax.legend(loc='upper left')
        
        # Plot Spectrogram (ctx_audio)
        ax_spec = axes[i, 1]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(ctx_audio, n_fft=512, hop_length=128)), ref=np.max)
        time_spec = np.linspace(-150, 50, D.shape[1])
        ax_spec.pcolormesh(time_spec, librosa.fft_frequencies(sr=24000, n_fft=512), D, shading='gouraud', cmap='viridis')
        
        ax_spec.axvline(x=burst_time_ms, color='white', linestyle='--', label='Detected Burst')
        ax_spec.axvline(x=0, color='red', linestyle='-', label='Vowel Anchor')
        ax_spec.axvspan(burst_time_ms - 20, burst_time_ms + 40, color='green', alpha=0.2, label='New Window')
        
        ax_spec.set_title(f"Spectrogram")
        ax_spec.set_xlim([-150, 50])
        ax_spec.set_ylim([0, 8000])

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'burst_detection_test.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()
