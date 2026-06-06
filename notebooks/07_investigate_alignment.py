import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.jvs_dataset import JVSCVExtractor

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jvs_ver1'))
    lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools/jvs_r9y9/aligned_labels_julius'))
    
    extractor = JVSCVExtractor(data_dir=data_dir, lab_dir=lab_dir, sr_orig=24000, sr_deg=8000)
    
    speakers = ['jvs001', 'jvs002']
    target_consonants = ['s', 'h', 'm', 'n']
    target_vowel = 'a'
    
    plotted = set()
    
    for speaker in speakers:
        lab_speaker_dir = os.path.join(lab_dir, speaker)
        if not os.path.exists(lab_speaker_dir): continue
            
        lab_files = sorted([f for f in os.listdir(lab_speaker_dir) if f.endswith('.lab')])
        
        for lab_file in lab_files:
            lab_path = os.path.join(lab_speaker_dir, lab_file)
            phonemes = extractor.parse_lab_file(lab_path)
            
            for i in range(len(phonemes) - 1):
                ph1 = phonemes[i]
                ph2 = phonemes[i+1]
                
                if ph1['phoneme'] in target_consonants and ph2['phoneme'] == target_vowel:
                    if ph1['phoneme'] not in plotted:
                        print(f"[{ph1['phoneme']}{target_vowel}] found in {speaker}/{lab_file}")
                        print("Context (from i-3 to i+3):")
                        start_idx = max(0, i - 3)
                        end_idx = min(len(phonemes), i + 4)
                        for j in range(start_idx, end_idx):
                            ph = phonemes[j]
                            marker = " <--" if j == i else ""
                            print(f"  {ph['start']:.3f} - {ph['end']:.3f} : {ph['phoneme']}{marker}")
                        print("-" * 40)
                        plotted.add(ph1['phoneme'])
            
            if len(plotted) == len(target_consonants):
                return

if __name__ == '__main__':
    main()
