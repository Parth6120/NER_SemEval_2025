"""
Module: predict_finetuned_mt.py
Purpose: Use fine-tuned MarianMT model to translate validation set and save predictions in specified structure.
"""
import os
import json
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def translate_sentences(sentences, model, tokenizer, batch_size=8):
    translations = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        translated = model.generate(**inputs)
        outputs = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations.extend(outputs)
    return translations

def main():
    # Paths
    val_path = r"E:\AISD\Term2\NLP\Project\NER_SemEval_2025\Data\validation\fr_FR.jsonl"
    model_dir = os.path.join(os.path.dirname(__file__), "..", "finetuned_placeholder_mt")
    output_dir = r"E:\AISD\Term2\NLP\Project\NER_SemEval_2025\Data\predictions\finetuned_placeholder_mt"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fr_FR.jsonl")

    # Load validation data
    val_data = load_jsonl(val_path)
    # Assume input English sentences are under 'text' or similar key
    if 'placeholder_sentence' in val_data[0]:
        src_texts = [ex['placeholder_sentence'] for ex in val_data]
    elif 'text' in val_data[0]:
        src_texts = [ex['text'] for ex in val_data]
    elif 'source' in val_data[0]:
        src_texts = [ex['source'] for ex in val_data]
    else:
        raise ValueError("Could not find source text key in validation data.")

    # Load model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Translate
    translations = translate_sentences(src_texts, model, tokenizer)

    # Save predictions in same JSONL structure, add 'prediction' key
    for ex, pred in zip(val_data, translations):
        ex['prediction'] = pred
    save_jsonl(val_data, output_path)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    import torch
    main()
