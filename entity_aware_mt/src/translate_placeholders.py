"""
Module: translate_placeholders.py
Purpose: Translate placeholder-injected English sentences to French using Helsinki-NLP/opus-mt-en-fr
"""
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import os

def load_data(csv_path):
    """Loads the placeholder-injected CSV data."""
    return pd.read_csv(csv_path)

def translate_sentences(sentences, model_name="Helsinki-NLP/opus-mt-en-fr", batch_size=8):
    """
    Translates a list of English sentences to French using a pre-trained model.
    Args:
        sentences (list): List of English sentences.
        model_name (str): Hugging Face model name.
        batch_size (int): Number of sentences per batch.
    Returns:
        list: Translated French sentences.
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        outputs = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations.extend(outputs)
    return translations

def main():
    # Path to the placeholder-injected data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "entity_placeholders.csv")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    # Translate placeholder-injected sentences
    print("Translating placeholder-injected sentences...")
    df["mt_placeholder"] = translate_sentences(df["placeholder_sentence"].tolist())
    
    # Save the results
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "entity_placeholders_translated.csv")
    df.to_csv(output_path, index=False)
    print(f"Entity-aware translations saved to {output_path}")

if __name__ == "__main__":
    main()
