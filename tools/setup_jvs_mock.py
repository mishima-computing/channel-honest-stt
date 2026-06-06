import os
import numpy as np
import scipy.io.wavfile as wavfile
import sys

def create_mock_jvs(speakers=['jvs001', 'jvs002'], num_files=5, sr=24000):
    """
    Creates dummy .wav files based on the actual duration specified in the .lab files.
    This allows the CI/CD pipeline to test the CV syllable extraction logic end-to-end
    without downloading the 30GB JVS corpus.
    """
    base_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'jvs_r9y9/aligned_labels_julius'))
    
    for speaker in speakers:
        out_dir = os.path.join(base_out_dir, speaker, 'parallel100', 'wav24kHz16bit')
        os.makedirs(out_dir, exist_ok=True)
        
        lab_dir = os.path.join(lab_base_dir, speaker)
        if not os.path.exists(lab_dir):
            print(f"Label directory not found for {speaker}: {lab_dir}")
            continue
            
        lab_files = [f for f in os.listdir(lab_dir) if f.startswith('VOICEACTRESS100_') and f.endswith('.lab')]
        lab_files.sort()
        
        for lab_file in lab_files[:num_files]:
            lab_path = os.path.join(lab_dir, lab_file)
            
            # Read the last line to get the total duration in seconds
            try:
                with open(lab_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    last_line = lines[-1].strip()
                    end_time = float(last_line.split()[1])
            except Exception as e:
                print(f"Error reading {lab_path}: {e}")
                continue
                
            # Create a mock waveform of exactly 'end_time' duration
            # Use a mix of white noise and a simple sine wave so it's not silent
            num_samples = int(end_time * sr)
            t = np.linspace(0, end_time, num_samples, endpoint=False)
            audio = 0.5 * np.random.randn(num_samples) + 0.5 * np.sin(2 * np.pi * 440 * t)
            
            # Normalize to 16-bit PCM
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            wav_name = lab_file.replace('.lab', '.wav')
            wav_path = os.path.join(out_dir, wav_name)
            
            wavfile.write(wav_path, sr, audio_int16)
            print(f"Created mock audio: {wav_path} ({end_time:.2f}s)")

if __name__ == '__main__':
    print("Setting up Mock JVS subset for CI/CD testing...")
    create_mock_jvs(speakers=['jvs001', 'jvs002'], num_files=20)
    print("Done!")
