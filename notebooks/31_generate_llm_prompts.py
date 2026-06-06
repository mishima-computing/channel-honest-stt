import os
import pandas as pd

def generate_prompts_from_csv(csv_path, threshold, output_path):
    df = pd.read_csv(csv_path)
    
    # Sort just in case
    df = df.sort_values(['sentence_id', 'token_idx'])
    
    prompts = {}
    
    for _, row in df.iterrows():
        sid = row['sentence_id']
        if sid not in prompts:
            prompts[sid] = []
            
        pred = row['pred_label']
        prob = row['max_prob']
        is_vowel = row['is_vowel']
        
        # Format the token based on type
        if is_vowel:
            token_str = pred.lower()
        else:
            token_str = f"<{pred}>"
            
        # Append '?' if below confidence threshold
        if prob < threshold:
            if is_vowel:
                token_str += "?"
            else:
                token_str = f"<{pred}?>"
                
        prompts[sid].append(token_str)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (sid, tokens) in enumerate(prompts.items()):
            sentence_str = " ".join(tokens)
            f.write(f"[Sentence {i+1} | ID: {sid}]\nMasked: {sentence_str}\n\n")

def main():
    import sys
    base_dir = os.path.dirname(__file__)
    
    files_to_process = ['e2e_predictions_10dB.csv', 'e2e_predictions_0dB.csv']
    
    for filename in files_to_process:
        csv_path = os.path.join(base_dir, filename)
        if not os.path.exists(csv_path):
            print(f"File {csv_path} not found.")
            continue
            
        base_name = os.path.splitext(filename)[0]
        
        # Generate for T=0.7
        out_07 = os.path.join(base_dir, f'{base_name}_prompts_T0.7.txt')
        generate_prompts_from_csv(csv_path, 0.7, out_07)
        print(f"Saved T=0.7 prompts to {out_07}")
        
        # Generate for T=0.85
        out_085 = os.path.join(base_dir, f'{base_name}_prompts_T0.85.txt')
        generate_prompts_from_csv(csv_path, 0.85, out_085)
        print(f"Saved T=0.85 prompts to {out_085}")

if __name__ == '__main__':
    main()
