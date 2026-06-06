import numpy as np
import scipy.signal as signal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.synthesizer.formant import FormantSynthesizer
from src.synthesizer.pipeline import DegradationPipeline

def get_peak_harmonic(fft_val, freqs, target_f, search_radius=150):
    mask = (freqs >= target_f - search_radius) & (freqs <= target_f + search_radius)
    if not np.any(mask): return 0, -np.inf
    peak_idx = np.argmax(fft_val[mask])
    return freqs[mask][peak_idx], fft_val[mask][peak_idx]

def get_valley_level(fft_val, freqs, f1, f2):
    # F1とF2の間の谷（F1+150Hz から F2-150Hz の範囲）
    start_f = f1 + 150
    end_f = f2 - 150
    if start_f >= end_f:
        # F1とF2が近すぎる場合（/o/の500と900など）、中点付近を取る
        mid = (f1 + f2) / 2
        start_f = mid - 50
        end_f = mid + 50
        
    mask = (freqs >= start_f) & (freqs <= end_f)
    if not np.any(mask): return -np.inf
    # 倍音の谷間を拾いすぎないよう、平均レベル（平均パワーのdB）を谷のレベルとする
    mean_power = np.mean(10 ** (fft_val[mask] / 10))
    return 10 * np.log10(mean_power + 1e-10)

def main():
    synth = FormantSynthesizer(48000)
    pipe = DegradationPipeline(8000)

    vowels = {
        '/i/': [(300, 50), (2300, 100), (3000, 120)],
        '/e/': [(450, 60), (2000, 100), (2800, 120)],
        '/a/': [(700, 80), (1200, 100), (2600, 120)],
        '/o/': [(500, 60), (900, 100), (2400, 120)],
        '/u/': [(350, 50), (1300, 100), (2500, 120)],
        '/e-i/': [(370, 55), (2150, 100), (2900, 120)],
        '/u-o/': [(420, 55), (1100, 100), (2450, 120)]
    }

    print("| 母音 | C_F2(Hz) | D_F2(Hz) | C_谷(dB) | D_谷(dB) | 谷の縮小(dB) | 判定 |")
    print("|---|---|---|---|---|---|---|")

    for name, formants in vowels.items():
        # 長めの0.5秒全体でFFTを取り、安定したスペクトルを得る
        t, y = synth.synthesize_vowel(120, formants, 0.5)
        y_clean = signal.resample_poly(y, 8000, 48000)
        y_deg = pipe.process(y, 48000, hpf_cutoff=500.0)
        
        # ハミング窓を適用
        y_c_win = y_clean * np.hamming(len(y_clean))
        y_d_win = y_deg * np.hamming(len(y_deg))
        
        # パワースペクトル(dB)
        fft_c = 20 * np.log10(np.abs(np.fft.rfft(y_c_win)) + 1e-10)
        fft_d = 20 * np.log10(np.abs(np.fft.rfft(y_d_win)) + 1e-10)
        freqs = np.fft.rfftfreq(len(y_clean), 1.0/8000)
        
        # Clean
        c_f1_hz, c_f1_db = get_peak_harmonic(fft_c, freqs, formants[0][0])
        c_f2_hz, c_f2_db = get_peak_harmonic(fft_c, freqs, formants[1][0])
        c_val_db = get_valley_level(fft_c, freqs, c_f1_hz, c_f2_hz)
        c_contrast = min(c_f1_db, c_f2_db) - c_val_db
        
        # Degraded
        d_f1_hz, d_f1_db = get_peak_harmonic(fft_d, freqs, formants[0][0])
        d_f2_hz, d_f2_db = get_peak_harmonic(fft_d, freqs, formants[1][0])
        d_val_db = get_valley_level(fft_d, freqs, d_f1_hz, d_f2_hz)
        d_contrast = min(d_f1_db, d_f2_db) - d_val_db
        
        # F1ピークが著しく減衰している場合（特に閉口母音のDegraded）は、F1は信頼できない
        # 500HzHPFによりF1が谷より低くなる場合があるため、その場合はF2と谷のコントラストを取る
        if d_f1_db < d_val_db + 3:
            d_contrast = d_f2_db - d_val_db
            
        if c_f1_db < c_val_db + 3:
            c_contrast = c_f2_db - c_val_db
            
        shrinkage = c_contrast - d_contrast
        
        # 判定基準（例：谷の深さが3dB未満なら「畳む」、それ以上なら「残す」）
        # ※実際には「畳む/残す」は母音ペアに適用するものだが、まずは個々の母音の分離度を見る
        judgment = "融合" if d_contrast < 3.0 else ("残存" if d_contrast > 8.0 else "要確認")
        
        print(f"| {name:<5} | {c_f2_hz:<8.0f} | {d_f2_hz:<8.0f} | {c_contrast:<8.1f} | {d_contrast:<8.1f} | {-shrinkage:<8.1f} | {judgment} |")

if __name__ == '__main__':
    main()
