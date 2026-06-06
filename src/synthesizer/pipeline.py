import numpy as np
from scipy import signal

class DegradationPipeline:
    """
    広帯域のクリーンな音声信号を、電話特有の劣悪な回線・スピーカー環境（ナロー帯域）を
    シミュレートして劣化させるDSPパイプライン。
    """
    def __init__(self, target_sr=8000):
        self.target_sr = target_sr

    def downsample(self, x, orig_sr):
        """
        [1] 8kHzへのダウンサンプリング
        """
        if orig_sr == self.target_sr:
            return x
        # 簡易的なリサンプリング。実運用ではアンチエイリアシングフィルタを確実に効かせる。
        num_samples = int(len(x) * float(self.target_sr) / orig_sr)
        return signal.resample(x, num_samples)

    def _ulaw_compress(self, x, mu=255.0):
        """
        μ-law 圧縮則に基づく非線形量子化（エンコード・デコードのシミュレーション）。
        G.711 μ-law コーデックによる量子化ノイズの付加を目的とする。
        x は -1.0 から 1.0 の範囲を想定。
        """
        # Encode (Continuous to 8-bit float equivalent)
        x_encoded = np.sign(x) * (np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu))
        
        # 8-bit 量子化のシミュレート（256階調）
        quantized = np.round(x_encoded * 128.0) / 128.0
        
        # Decode
        x_decoded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu)**np.abs(quantized) - 1.0)
        return x_decoded

    def apply_codec(self, x, codec_type='ulaw'):
        """
        [2] 回線コーデックの適用
        Phase 1: 'ulaw' (G.711 μ-law) 
        Phase 2: 'amr-nb' を将来的に追加予定。
        """
        if codec_type == 'ulaw':
            return self._ulaw_compress(x)
        elif codec_type == 'amr-nb':
            raise NotImplementedError("AMR-NB codec simulation is not yet implemented.")
        else:
            raise ValueError(f"Unknown codec_type: {codec_type}")

    def apply_speaker_filter(self, x, cutoff=500.0, order=2):
        """
        [3] 受話側スピーカーフィルタ（低域カット）
        電話機の安価なスピーカーによる低域欠落を再現する。
        - 500Hz基準のハイパスフィルタ
        - 2次（-12dB/oct）の緩やかなロールオフ
        """
        nyq = 0.5 * self.target_sr
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.lfilter(b, a, x)

    def apply_clipping(self, x, drive=1.5):
        """
        [4] 非線形歪み
        ソフトクリッピング（tanh）により、大音量時のスピーカー歪みを再現する。
        """
        return np.tanh(drive * x)

    def process(self, x, orig_sr, codec_type='ulaw', hpf_cutoff=500.0, drive=1.5):
        """
        議事録の「信号の物理順」に従って全劣化チェーンを適用する。
        """
        # 1. ダウンサンプル
        x_down = self.downsample(x, orig_sr)
        
        # 2. 回線コーデック
        x_codec = self.apply_codec(x_down, codec_type=codec_type)
        
        # 3. スピーカーフィルタ（低域カット）
        x_filtered = self.apply_speaker_filter(x_codec, cutoff=hpf_cutoff)
        
        # 4. 非線形歪み（クリッピング）
        x_out = self.apply_clipping(x_filtered, drive=drive)
        
        return x_out

if __name__ == "__main__":
    # 簡易テスト
    pipeline = DegradationPipeline()
    test_signal = np.sin(2 * np.pi * 300 * np.linspace(0, 1, 48000))
    out = pipeline.process(test_signal, orig_sr=48000)
    print(f"Pipeline output shape: {out.shape}, max: {out.max():.2f}")
