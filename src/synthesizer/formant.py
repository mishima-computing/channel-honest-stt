import numpy as np
from scipy import signal

class FormantSynthesizer:
    """
    シンプルなソース・フィルタモデルによるフォルマント合成器。
    指定された基本周波数(f0)と、複数のフォルマント周波数(F1, F2, F3...)を用いて
    母音のような音声を合成する。
    """
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    def generate_impulse_train(self, f0, duration):
        """
        声帯音源に相当する単純なインパルス列（パルス波）を生成する。
        """
        t = np.arange(0, duration, 1.0 / self.sample_rate)
        pulse_period = 1.0 / f0
        
        # 単純なインパルス列
        source = np.zeros_like(t)
        pulse_indices = np.arange(0, len(t), int(self.sample_rate * pulse_period)).astype(int)
        source[pulse_indices] = 1.0
        
        return t, source

    def apply_formant_filter(self, x, f_center, bandwidth):
        """
        単一のフォルマント（共振）フィルタを適用する（2nd order IIR peak filter）。
        """
        Q = f_center / bandwidth
        b, a = signal.iirpeak(f_center, Q, self.sample_rate)
        return signal.lfilter(b, a, x)

    def synthesize_vowel(self, f0, formants, duration=1.0):
        """
        指定されたフォルマント群から母音を合成する。
        formants: [(f_center1, bw1), (f_center2, bw2), ...] のリスト
        """
        t, source = self.generate_impulse_train(f0, duration)
        
        # 直列（カスケード）接続でフィルタリング
        y = source.copy()
        for f_center, bw in formants:
            y = self.apply_formant_filter(y, f_center, bw)
            
        # 波形の正規化 (-1.0 to 1.0)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
            
        return t, y

if __name__ == "__main__":
    # テスト用: 母音 /i/ の生成
    synth = FormantSynthesizer(sample_rate=48000)
    # /i/の典型的なフォルマント (F1:300Hz, F2:2300Hz, F3:3000Hz)
    formants = [(300, 50), (2300, 100), (3000, 120)]
    t, y = synth.synthesize_vowel(f0=120, formants=formants, duration=0.5)
    print(f"Generated shape: {y.shape}, max: {y.max():.2f}, min: {y.min():.2f}")
