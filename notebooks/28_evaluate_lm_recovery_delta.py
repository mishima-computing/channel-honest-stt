import os
import pyopenjtalk
import Levenshtein

def evaluate_delta(answer_file, pred_file):
    if not os.path.exists(answer_file) or not os.path.exists(pred_file):
        print("Missing answer or prediction files.")
        return

    original_sentences = []
    original_phonemes_list = []
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Original: '):
                original_sentences.append(line.replace('Original: ', '').strip())
            elif line.startswith('Phonemes: '):
                original_phonemes_list.append(line.replace('Phonemes: ', '').strip().split(' '))

    predicted_sentences = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predicted_sentences.append(line)

    if len(original_sentences) != len(predicted_sentences):
        print("Count mismatch between originals and predictions!")
        return

    manner_map = {
        'p': 'Unvoiced_Plosive', 'py': 'Unvoiced_Plosive', 't': 'Unvoiced_Plosive', 'k': 'Unvoiced_Plosive', 'ky': 'Unvoiced_Plosive',
        'ts': 'Unvoiced_Plosive', 'ch': 'Unvoiced_Plosive', 
        'b': 'Voiced_Plosive', 'by': 'Voiced_Plosive', 'd': 'Voiced_Plosive', 'g': 'Voiced_Plosive', 'gy': 'Voiced_Plosive',
        'z': 'Voiced_Plosive', 'j': 'Voiced_Plosive', 'r': 'Voiced_Plosive', 'ry': 'Voiced_Plosive',
        's': 'Unvoiced_Fricative', 'sh': 'Unvoiced_Fricative', 'h': 'Unvoiced_Fricative', 'hy': 'Unvoiced_Fricative', 'f': 'Unvoiced_Fricative',
        'm': 'Nasal', 'my': 'Nasal', 'n': 'Nasal', 'ny': 'Nasal'
    }

    stats = {
        'Unvoiced_Plosive': {'total': 0, 'correct': 0},
        'Voiced_Plosive': {'total': 0, 'correct': 0},
        'Unvoiced_Fricative': {'total': 0, 'correct': 0},
        'Nasal': {'total': 0, 'correct': 0}
    }

    for orig_text, orig_phons, pred_text in zip(original_sentences, original_phonemes_list, predicted_sentences):
        pred_phons = pyopenjtalk.g2p(pred_text).split(' ')
        
        # Align using Levenshtein
        # We need a string representation to align sequences, but we have lists.
        # So we map each phoneme to a unique char for alignment.
        vocab = list(set(orig_phons + pred_phons))
        v2c = {v: chr(i+1000) for i, v in enumerate(vocab)}
        
        orig_str = "".join([v2c[p] for p in orig_phons])
        pred_str = "".join([v2c[p] for p in pred_phons])
        
        ops = Levenshtein.editops(orig_str, pred_str)
        
        # apply ops to track correctness
        aligned_pred = [None] * len(orig_phons)
        
        # Initialize with perfect match
        for i in range(len(orig_phons)):
            aligned_pred[i] = orig_phons[i]
            
        for op, o_idx, p_idx in ops:
            if op == 'replace':
                aligned_pred[o_idx] = pred_phons[p_idx]
            elif op == 'delete':
                aligned_pred[o_idx] = '<DELETED>'
            # insertions in pred are ignored because they don't map to an original phoneme
            
        for o_p, a_p in zip(orig_phons, aligned_pred):
            if o_p in manner_map:
                manner = manner_map[o_p]
                stats[manner]['total'] += 1
                if o_p == a_p:
                    stats[manner]['correct'] += 1

    print("=== LM Recovery Delta Breakdown ===")
    print("Class                | Correct / Total | Accuracy")
    print("-" * 50)
    for manner, data in stats.items():
        total = data['total']
        correct = data['correct']
        acc = correct / total if total > 0 else 0
        print(f"{manner:20} | {correct:7} / {total:<5} | {acc:.2%}")
        
    print("\n※ Note: Delta = (Clean Upper Bound - Above Accuracy)")

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    ans_path = os.path.join(base_dir, 'blind_answers.txt')
    pred_path = os.path.join(base_dir, 'llm_predictions.txt')
    evaluate_delta(ans_path, pred_path)
