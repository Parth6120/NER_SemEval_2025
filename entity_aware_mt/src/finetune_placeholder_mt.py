"""
Module: finetune_placeholder_mt.py
Purpose: Fine-tune Helsinki-NLP/opus-mt-en-fr on placeholder-injected English → placeholder-injected French sentence pairs
"""
import os
import pandas as pd
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Use placeholder-injected English as source, and placeholder-injected French as target
    # For training, you need to have both. If you don't have placeholder-injected French, use the original French with placeholders inserted at the same positions as in English.
    # Here, we assume you have a column 'placeholder_sentence' (English) and 'target_placeholder' (French)
    # If not, you may need to generate 'target_placeholder' first.
    if 'target_placeholder' not in df.columns:
        # Fallback: use 'target' (reference French) for now
        df['target_placeholder'] = df['target']
    return df[['placeholder_sentence', 'target_placeholder']]

def preprocess_function(examples, tokenizer, max_length=128):
    model_inputs = tokenizer(examples['placeholder_sentence'], max_length=max_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_placeholder'], max_length=max_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "entity_placeholders.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "finetuned_placeholder_mt")
    batch_size = 8
    num_train_epochs = 3
    max_length = 128

    # Load data
    df = load_data(data_path)
    dataset = Dataset.from_pandas(df)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Preprocess
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="no",
        logging_steps=100,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()
