import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 親ディレクトリのモジュールを読み込めるようにする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthesizer.formant import FormantSynthesizer
from src.synthesizer.pipeline import DegradationPipeline

def plot_spectrogram(ax, y, sr, title):
    ax.specgram(y, Fs=sr, NFFT=256, noverlap=128, cmap='magma')
    ax.set_title(title)
    ax.set_ylabel("Freq (Hz)")
    ax.set_ylim(0, 4000)

def main():
    print("=== 劣化チェーン合成器 目視検証 ===")
    
    orig_sr = 48000
    duration = 0.5
    f0 = 120 # 成人男性F0 (このパラメータを可変にして話者正規化検証に使う)
    
    synth = FormantSynthesizer(sample_rate=orig_sr)
    pipeline = DegradationPipeline(target_sr=8000)
    
    # --- 1. /i/ の検証 (500Hzカットの確認) ---
    print("Generating /i/ (F1=300, F2=2300)...")
    _, y_i = synth.synthesize_vowel(f0=f0, formants=[(300, 50), (2300, 100), (3000, 120)], duration=duration)
    y_i_deg = pipeline.process(y_i, orig_sr, codec_type='ulaw', hpf_cutoff=500.0, drive=2.0)
    
    # --- 2. /u/ vs /o/ の検証 (混ざる井戸対の確認) ---
    print("Generating /u/ (F1=350, F2=1300) and /o/ (F1=500, F2=900)...")
    _, y_u = synth.synthesize_vowel(f0=f0, formants=[(350, 50), (1300, 80), (2500, 100)], duration=duration)
    _, y_o = synth.synthesize_vowel(f0=f0, formants=[(500, 60), (900, 80), (2400, 100)], duration=duration)
    
    y_u_deg = pipeline.process(y_u, orig_sr)
    y_o_deg = pipeline.process(y_o, orig_sr)

    # --- 3. /s/ vs /h/ の検証 (無声摩擦音の確認) ---
    print("Generating /s/ (High freq noise) and /h/ (Formant noise)...")
    _, noise = synth.generate_noise_source(duration)
    # /s/ は 4000Hz以上にエネルギーが集中するが、ナロー帯域ではどうなるか
    y_s = synth.apply_formant_filter(noise, 4500, 1000) 
    # /h/ は声道全体のフォルマントが乗る
    y_h = synth.apply_formant_filter(noise, 1000, 500)
    
    y_s_deg = pipeline.process(y_s, orig_sr)
    y_h_deg = pipeline.process(y_h, orig_sr)
    
    # プロット
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    plot_spectrogram(axes[0, 0], pipeline.downsample(y_i, orig_sr), 8000, "/i/ Clean (8kHz)")
    plot_spectrogram(axes[0, 1], y_i_deg, 8000, "/i/ Degraded (F1 300Hz attenuated)")
    
    plot_spectrogram(axes[1, 0], y_u_deg, 8000, "/u/ Degraded (F2=1300Hz)")
    plot_spectrogram(axes[1, 1], y_o_deg, 8000, "/o/ Degraded (F2=900Hz)")
    
    plot_spectrogram(axes[2, 0], y_s_deg, 8000, "/s/ Degraded (High Freq cut off)")
    plot_spectrogram(axes[2, 1], y_h_deg, 8000, "/h/ Degraded (Broadband noise)")
    
    plt.tight_layout()
    output_path = "spectrogram_check.png"
    plt.savefig(output_path)
    print(f"Done! Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
