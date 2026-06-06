import numpy as np
import scipy.signal as signal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.synthesizer.formant import FormantSynthesizer
from src.synthesizer.pipeline import DegradationPipeline

def get_peak_harmonic(y, target_f, sr=8000, search_radius=150):
    """
    指定されたターゲット周波数付近の最も強いピーク（倍音）成分を抽出する
    """
    y_win = y * np.hamming(len(y))
    fft_val = np.abs(np.fft.rfft(y_win))
    freqs = np.fft.rfftfreq(len(y), 1.0/sr)
    mask = (freqs >= target_f - search_radius) & (freqs <= target_f + search_radius)
    if not np.any(mask): return 0
    return freqs[mask][np.argmax(fft_val[mask])]

def get_formants_lpc(y, sr, order=12):
    import scipy.linalg
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    y = y * np.hamming(len(y))
    r = np.correlate(y, y, mode='full')
    r = r[len(y)-1:len(y)+order]
    a = scipy.linalg.solve_toeplitz((r[:-1], r[:-1]), -r[1:])
    lpc_coefs = np.concatenate(([1], a))
    roots = np.roots(lpc_coefs)
    roots = roots[np.imag(roots) > 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = sorted(angles * (sr / (2 * np.pi)))
    formants = [f for f in freqs if f > 150]
    return formants[:2] if len(formants) >= 2 else (formants + [0, 0])[:2]

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

    print("| Vowel | Clean F1 | Clean F2 | Degraded F1 | Degraded F2 | F1 Shift | F2 Shift |")
    print("|-------|----------|----------|-------------|-------------|----------|----------|")

    for name, formants in vowels.items():
        t, y = synth.synthesize_vowel(120, formants, 0.5)
        y_clean = signal.resample_poly(y, 8000, 48000)
        y_deg = pipe.process(y, 48000, hpf_cutoff=500.0)
        
        c_f1 = get_peak_harmonic(y_clean, formants[0][0])
        c_f2 = get_peak_harmonic(y_clean, formants[1][0])
        d_f1 = get_peak_harmonic(y_deg, formants[0][0])
        d_f2 = get_peak_harmonic(y_deg, formants[1][0])
        
        s_f1 = d_f1 - c_f1
        s_f2 = d_f2 - c_f2
        
        print(f"| {name:<5} | {c_f1:<8.0f} | {c_f2:<8.0f} | {d_f1:<11.0f} | {d_f2:<11.0f} | {s_f1:<8.0f} | {s_f2:<8.0f} |")

if __name__ == '__main__':
    main()
