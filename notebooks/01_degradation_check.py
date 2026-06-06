import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 親ディレクトリのモジュールを読み込めるようにする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthesizer.formant import FormantSynthesizer
from src.synthesizer.pipeline import DegradationPipeline

def plot_spectrogram(ax, y, sr, title):
    """
    指定されたAxesにスペクトログラムを描画する。
    """
    ax.specgram(y, Fs=sr, NFFT=256, noverlap=128, cmap='magma')
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 4000) # ナロー帯域の範囲(0-4kHz)にフォーカス

def main():
    print("=== 劣化チェーン合成器 目視検証 ===")
    
    # 1. クリーンな広帯域母音 /i/ の生成
    # /i/ は F1 が極端に低い (300Hz付近) ため、500Hzハイパスの検証に最適
    orig_sr = 48000
    f0 = 120 # 成人男性の典型的な基本周波数
    formants = [(300, 50), (2300, 100), (3000, 120)]
    duration = 0.5
    
    print(f"Generating clean vowel /i/ (F1={formants[0][0]}Hz)...")
    synth = FormantSynthesizer(sample_rate=orig_sr)
    t, y_clean = synth.synthesize_vowel(f0=f0, formants=formants, duration=duration)
    
    # 2. 劣化パイプラインの適用
    print("Applying degradation pipeline...")
    pipeline = DegradationPipeline(target_sr=8000)
    
    # 各段の出力を個別に取得して確認する
    # A. ダウンサンプリングのみ
    y_down = pipeline.downsample(y_clean, orig_sr)
    
    # B. ダウンサンプル + μ-law コーデック
    y_codec = pipeline.apply_codec(y_down, codec_type='ulaw')
    
    # C. ダウンサンプル + μ-law + スピーカーHPF(500Hz)
    y_hpf = pipeline.apply_speaker_filter(y_codec, cutoff=500.0)
    
    # D. 全て ( + tanhクリッピング )
    y_final = pipeline.process(y_clean, orig_sr, codec_type='ulaw', hpf_cutoff=500.0, drive=2.0)
    
    # 3. スペクトログラムの描画
    print("Generating spectrograms...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    plot_spectrogram(axes[0, 0], y_down, 8000, "1. Clean (8kHz Downsampled)")
    plot_spectrogram(axes[0, 1], y_codec, 8000, "2. + G.711 mu-law (Quantization Noise)")
    plot_spectrogram(axes[1, 0], y_hpf, 8000, "3. + 500Hz HPF (F1 300Hz is attenuated)")
    plot_spectrogram(axes[1, 1], y_final, 8000, "4. + Tanh Soft Clipping (Final)")
    
    plt.tight_layout()
    output_path = "spectrogram_check.png"
    plt.savefig(output_path)
    print(f"Done! Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
