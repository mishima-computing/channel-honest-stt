import os
import sys
from openai import OpenAI

def get_system_prompt():
    return """これは日本語音声認識の曖昧性解消タスクです。
電話回線の雑音により、各音節の子音の「調音位置」が不明になっています。
ただし以下の情報は判明しています：
- 子音の種類（無声破裂/有声破裂/無声摩擦/鼻音のいずれか）
- 各音節の母音（a/i/u/e/o）
タグの意味：
- <Unvoiced_Plosive> = k,t,p,ts,ch のいずれか（無声・破裂/破擦）
- <Voiced_Plosive> = g,d,b,z,j,r のいずれか（有声・破裂/破擦/摩擦/弾き）
- <Unvoiced_Fricative> = s,sh,h,f のいずれか（無声・摩擦）
- <Nasal> = m,n のいずれか（鼻音）
- 末尾に ? が付くタグは「分類器自身が不確実と判断した箇所」（特に注意）
これらを手がかりに、文脈から最も自然な日本語の文章を推定してください。
出力は復元した自然な日本語テキスト（漢字・かな交じり）のみ。説明や前置き、読みがな（音素列）は一切不要です。

【例】
Masked: k o <Nasal> n i <Unvoiced_Plosive> i w a
Output: こんにちは
"""

def load_sentences(filepath, limit=10):
    sentences = []
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return sentences
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    for b in blocks[:limit]:
        lines = b.split('\n')
        if len(lines) >= 2:
            sentence_id = lines[0]
            masked_text = lines[1].replace('Masked: ', '')
            sentences.append((sentence_id, masked_text))
            
    return sentences

def main():
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    
    # Check connection and get model name
    try:
        models = client.models.list()
        model_name = models.data[0].id
        print(f"Connected to LM Studio. Using model: {model_name}")
    except Exception as e:
        print(f"Failed to connect to LM Studio at 127.0.0.1:1234. Please ensure it is running and the local server is started.")
        print(f"Error: {e}")
        return

    base_dir = os.path.dirname(__file__)
    prompt_file = os.path.join(base_dir, 'e2e_predictions_10dB_prompts_T0.7.txt')
    
    sentences = load_sentences(prompt_file, limit=10)
    if not sentences:
        return
        
    print(f"\n--- Testing 10 sentences with {model_name} ---")
    
    results = []
    
    for i, (sid, masked) in enumerate(sentences):
        print(f"\n[{i+1}/10] {sid}")
        print(f"Masked: {masked}")
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": masked}
                ],
                temperature=0.0,
                max_tokens=150
            )
            
            output = response.choices[0].message.content.strip()
            print(f"Output: {output}")
            results.append((sid, masked, output))
            
        except Exception as e:
            print(f"API Error during inference: {e}")
            break
            
    # Save output for review
    out_file = os.path.join(base_dir, f'lm_studio_test_{model_name.replace("/", "_")}.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n\n")
        for sid, masked, output in results:
            f.write(f"{sid}\nMasked: {masked}\nOutput: {output}\n\n")
            
    print(f"\nResults saved to {out_file}")
    print("Please review the outputs for Japanese quality, instruction following, and format.")

if __name__ == "__main__":
    main()
