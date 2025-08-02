"""
Module: postprocess_entities.py
Purpose: Replace placeholders in translated French sentences with correct entity translations.
"""
import pandas as pd
import ast
import os
import requests

def get_label_from_wikidata(qid, lang='fr'):
    """Fetches the label for a given Wikidata QID in the specified language."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['entities'][qid]['labels'][lang]['value']
    except Exception as e:
        print(f"Could not fetch label for {qid}: {e}")
        return None

def replace_placeholders(row):
    sentence = row['mt_placeholder']
    placeholder_map = ast.literal_eval(row['placeholder_map']) if isinstance(row['placeholder_map'], str) else row['placeholder_map']
    for placeholder, qid in placeholder_map.items():
        if qid:
            fr_label = get_label_from_wikidata(qid, lang='fr')
            if not fr_label:
                fr_label = qid  # fallback to QID if label not found
            sentence = sentence.replace(placeholder, fr_label)
    row['mt_entity_aware'] = sentence
    return row

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "entity_placeholders_translated.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    print("Replacing placeholders with French entity labels...")
    df = df.apply(replace_placeholders, axis=1)

    # Save the final entity-aware translations
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "entity_aware_translations.csv")
    df.to_csv(output_path, index=False)
    print(f"Entity-aware translations saved to {output_path}")

if __name__ == "__main__":
    main()
