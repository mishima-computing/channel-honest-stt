import os
import re
import Levenshtein

def evaluate_recovery(original_texts, predicted_texts):
    if len(original_texts) != len(predicted_texts):
        print(f"Warning: Count mismatch. {len(original_texts)} originals vs {len(predicted_texts)} predictions.")
    
    total_cer = 0.0
    exact_matches = 0
    
    for orig, pred in zip(original_texts, predicted_texts):
        orig_clean = re.sub(r'[\s、。！？]', '', orig)
        pred_clean = re.sub(r'[\s、。！？]', '', pred)
        
        dist = Levenshtein.distance(orig_clean, pred_clean)
        cer = dist / len(orig_clean) if len(orig_clean) > 0 else 0
        total_cer += cer
        
        if dist == 0:
            exact_matches += 1
            
    avg_cer = total_cer / len(original_texts) if original_texts else 0
    accuracy = exact_matches / len(original_texts) if original_texts else 0
    
    return accuracy, avg_cer

def main():
    # This is a template for evaluating LLM recovery.
    # Users will generate the prompts using 24_generate_lm_recovery_dataset.py,
    # pass them to an LLM, and save the LLM's decoded sentences to 'llm_predictions.txt'
    
    answer_file = os.path.join(os.path.dirname(__file__), 'blind_answers.txt')
    pred_file = os.path.join(os.path.dirname(__file__), 'llm_predictions.txt')
    
    if not os.path.exists(answer_file):
        print(f"Answer file not found: {answer_file}")
        return
        
    original_sentences = []
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Original: '):
                original_sentences.append(line.replace('Original: ', '').strip())
                
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        print("Please run the LLM on 'blind_prompts.txt' and save one predicted sentence per line in 'llm_predictions.txt'.")
        return
        
    predicted_sentences = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predicted_sentences.append(line)
                
    acc, cer = evaluate_recovery(original_sentences, predicted_sentences)
    
    print("=== LM Recovery Evaluation ===")
    print(f"Total Sentences: {len(original_sentences)}")
    print(f"Exact Sentence Match Accuracy: {acc:.2%}")
    print(f"Character Error Rate (CER): {cer:.2%}")
    
    # Delta logic would require running it without masking and with masking,
    # and subtracting the CERs.
    print("\nNote on Delta Evaluation:")
    print("To compute Delta per phonetic class, run the LLM twice:")
    print("1. Baseline: Feed the phonemes WITHOUT masking.")
    print("2. Masked: Feed the phonemes WITH masking (<Unvoiced_Plosive> etc).")
    print("Delta = (CER Masked - CER Baseline).")

if __name__ == '__main__':
    main()
