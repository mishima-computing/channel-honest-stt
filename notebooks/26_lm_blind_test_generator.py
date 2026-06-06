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
                
    random.seed(12345) # Different seed
    sample_sentences = random.sample(sentences, 10)
    
    manner_map = {
        'p': '<UV_PLOS>', 'py': '<UV_PLOS>', 't': '<UV_PLOS>', 'k': '<UV_PLOS>', 'ky': '<UV_PLOS>',
        'ts': '<UV_PLOS>', 'ch': '<UV_PLOS>', 
        'b': '<V_PLOS>', 'by': '<V_PLOS>', 'd': '<V_PLOS>', 'g': '<V_PLOS>', 'gy': '<V_PLOS>',
        'z': '<V_PLOS>', 'j': '<V_PLOS>', 'r': '<V_PLOS>', 'ry': '<V_PLOS>',
        's': '<UV_FRIC>', 'sh': '<UV_FRIC>', 'h': '<UV_FRIC>', 'hy': '<UV_FRIC>', 'f': '<UV_FRIC>',
        'm': '<NASAL>', 'my': '<NASAL>', 'n': '<NASAL>', 'ny': '<NASAL>'
    }
    
    prompt_file = os.path.join(os.path.dirname(__file__), 'blind_prompts.txt')
    answer_file = os.path.join(os.path.dirname(__file__), 'blind_answers.txt')
    
    with open(prompt_file, 'w', encoding='utf-8') as fp, open(answer_file, 'w', encoding='utf-8') as fa:
        fp.write("=== LM Blind Recovery Test ===\n\n")
        fa.write("=== LM Blind Recovery Answers ===\n\n")
        
        for i, sent in enumerate(sample_sentences):
            phonemes = pyopenjtalk.g2p(sent).split(' ')
            
            masked_phonemes = []
            for p in phonemes:
                if p in manner_map:
                    masked_phonemes.append(manner_map[p])
                else:
                    masked_phonemes.append(p)
                    
            fp.write(f"[Sentence {i+1}]\n")
            fp.write(f"Masked: {' '.join(masked_phonemes)}\n\n")
            
            fa.write(f"[Sentence {i+1}]\n")
            fa.write(f"Original: {sent}\n")
            fa.write(f"Phonemes: {' '.join(phonemes)}\n\n")
            
    print("Generated blind prompts and answers.")

if __name__ == '__main__':
    main()
