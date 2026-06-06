import os
import random
import pyopenjtalk

def main():
    transcript_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt'
    ))
    
    sentences = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                sentences.append(parts[1])
                
    random.seed(42)
    sample_sentences = random.sample(sentences, 10)
    
    # 9-class mapping
    manner_map = {
        'p': '<Unvoiced_Plosive>', 'py': '<Unvoiced_Plosive>', 't': '<Unvoiced_Plosive>', 'k': '<Unvoiced_Plosive>', 'ky': '<Unvoiced_Plosive>',
        'ts': '<Unvoiced_Plosive>', 'ch': '<Unvoiced_Plosive>', 
        'b': '<Voiced_Plosive>', 'by': '<Voiced_Plosive>', 'd': '<Voiced_Plosive>', 'g': '<Voiced_Plosive>', 'gy': '<Voiced_Plosive>',
        'z': '<Voiced_Plosive>', 'j': '<Voiced_Plosive>', 'r': '<Voiced_Plosive>', 'ry': '<Voiced_Plosive>',
        's': '<Unvoiced_Fricative>', 'sh': '<Unvoiced_Fricative>', 'h': '<Unvoiced_Fricative>', 'hy': '<Unvoiced_Fricative>', 'f': '<Unvoiced_Fricative>',
        'm': '<Nasal>', 'my': '<Nasal>', 'n': '<Nasal>', 'ny': '<Nasal>'
    }
    
    print("=== LM Recovery Test Dataset ===")
    print("Task: Recover the original Japanese text from the masked phoneme sequence.\n")
    
    for i, sent in enumerate(sample_sentences):
        phonemes = pyopenjtalk.g2p(sent).split(' ')
        
        masked_phonemes = []
        for p in phonemes:
            if p in manner_map:
                masked_phonemes.append(manner_map[p])
            else:
                masked_phonemes.append(p)
                
        print(f"[Test {i+1}]")
        print(f"Masked Phonemes:   {' '.join(masked_phonemes)}")
        print(f"Original Text (Hidden): {sent}")
        print(f"Original Phonemes (Hidden): {' '.join(phonemes)}\n")

if __name__ == '__main__':
    main()
