import pandas as pd
import requests
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def get_label_from_wikidata(qid):
    """Fetches the English label for a given Wikidata QID."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return data['entities'][qid]['labels']['en']['value']
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"Could not fetch label for {qid}: {e}")
        return None

def create_qid_to_label_mapping(df):
    """Creates a mapping from QID to its English label."""
    all_qids = set(qid for entity_list in df['entities'] for qid in entity_list)
    qid_to_label = {qid: get_label_from_wikidata(qid) for qid in all_qids}
    return qid_to_label

def tokenize_and_iob(row, qid_to_label):
    """Tokenizes source text and creates IOB tags for entities."""
    text = row['source']
    tokens = word_tokenize(text)
    labels = ['O'] * len(tokens)

    for qid in row['entities']:
        entity_text = qid_to_label.get(qid)
        if not entity_text:
            continue
        
        entity_tokens = word_tokenize(entity_text)
        if not entity_tokens:
            continue

        # Find entity in tokens and apply IOB tags
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                labels[i] = 'B-ENT'
                for j in range(1, len(entity_tokens)):
                    labels[i+j] = 'I-ENT'
                break  # Move to the next qid once tagged

    return list(zip(tokens, labels))

def prepare_data(file_path):
    """
    Loads data from a JSONL file and prepares it for NER and translation.
    
    Args:
        file_path (str): The path to the .jsonl file.
        
    Returns:
        pandas.DataFrame: A DataFrame with an added 'token_iob' column.
    """
    print("Loading data...")
    df = pd.read_json(file_path, lines=True)
    
    print("Fetching entity labels from Wikidata...")
    qid_to_label = create_qid_to_label_mapping(df)
    
    print("Tokenizing and creating IOB tags...")
    df['token_iob'] = df.apply(lambda row: tokenize_and_iob(row, qid_to_label), axis=1)
    
    print("Data preparation complete.")
    return df

if __name__ == '__main__':
    # Example usage:
    # Make sure to place your data file in the correct path
    train_file = r'E:\AISD\Term2\NLP\Project\NER_SemEval_2025\Data\train\fr\train.jsonl'
    
    prepared_df = prepare_data(train_file)
    
    # Display info and head of the processed DataFrame
    print("\nDataFrame Info:")
    prepared_df.info()
    print("\nFirst 5 rows of prepared data:")
    print(prepared_df.head())
    print("\nExample of token_iob column:")
    print(prepared_df['token_iob'].iloc[0])

    # Save the prepared DataFrame as a CSV file in the data folder
    output_path = r'E:\AISD\Term2\NLP\Project\NER_SemEval_2025\entity_aware_mt\data\prepared_data.csv'
    prepared_df.to_csv(output_path, index=False)
    print(f"\nPrepared data saved to {output_path}")
