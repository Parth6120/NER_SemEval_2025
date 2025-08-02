"""
Module: entity_aware_translation.py
Step 1: Placeholder injection for entity-aware MT
"""
import pandas as pd
import ast
import os

def extract_entity_spans(token_iob):
    """Extracts spans for entities using IOB tags."""
    spans = []
    current = None
    for idx, (token, tag) in enumerate(token_iob):
        if tag == 'B-ENT':
            if current:
                spans.append(current)
            current = [idx, idx]
        elif tag == 'I-ENT' and current:
            current[1] = idx
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans

def inject_placeholders(row):
    """Replaces entity spans with placeholders in the sentence."""
    token_iob = ast.literal_eval(row['token_iob'])
    tokens = [tok for tok, tag in token_iob]
    spans = extract_entity_spans(token_iob)
    entities = ast.literal_eval(row['entities'])
    placeholder_map = {}
    new_tokens = tokens[:]
    for idx, span in enumerate(spans):
        placeholder = f"@ENTITY{idx+1}@"
        # Replace entity tokens with placeholder
        start, end = span
        new_tokens[start:end+1] = [placeholder]
        placeholder_map[placeholder] = entities[idx] if idx < len(entities) else None
    row['placeholder_sentence'] = ' '.join(new_tokens)
    row['placeholder_map'] = placeholder_map
    return row

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "prepared_data.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    
    print("Injecting placeholders for entities...")
    df = df.apply(inject_placeholders, axis=1)
    
    # Save the intermediate result
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "entity_placeholders.csv")
    df.to_csv(output_path, index=False)
    print(f"Placeholder-injected data saved to {output_path}")

if __name__ == "__main__":
    main()
