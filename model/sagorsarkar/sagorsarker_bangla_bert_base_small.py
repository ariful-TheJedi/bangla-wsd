
import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# Configurations
JSON_DATA_PATH = "./new-data.json"
MODEL_NAME = "sagorsarker/bangla-bert-base"
MODEL_OUTPUT_DIR = "sagorsarker_bangla_bert_base"
TRAIN_ARGS = {
    "learning_rate": 3e-5,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "weight_decay": 0.01,
    "report_to": "none",
    "save_total_limit": 1
}

# Data Processor
class BengaliWSDDataProcessor:
    def __init__(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.examples = self._flatten_data()
        self._setup_label_mappings()

    def _flatten_data(self):
        flattened = []
        for entry in self.raw_data['data']:
            word = entry['word']
            for sense_info in entry['senses']:
                sense = sense_info['sense']
                for example in sense_info['examples']:
                    flattened.append({
                        'sentence': example,
                        'target_word': word,
                        'sense': sense
                    })
        return flattened

    def _setup_label_mappings(self):
        self.sense_labels = sorted({ex['sense'] for ex in self.examples})
        self.label2id = {label: i for i, label in enumerate(self.sense_labels)}
        self.id2label = {i: label for i, label in enumerate(self.sense_labels)}

    def get_dataset(self):
        return Dataset.from_dict({
            'sentence': [ex['sentence'] for ex in self.examples],
            'target_word': [ex['target_word'] for ex in self.examples],
            'sense': [ex['sense'] for ex in self.examples],
            'label': [self.label2id[ex['sense']] for ex in self.examples]
        })

# Tokenization Function
def tokenize_with_markers(batch, tokenizer):
    processed = {'input_ids': [], 'attention_mask': [], 'label': []}
    for sentence, word, label in zip(batch['sentence'], batch['target_word'], batch['label']):
        marked_sent = sentence.replace(word, f"[TGT]{word}[/TGT]")
        encoding = tokenizer(
            marked_sent,
            truncation=True,
            max_length=128,
            padding=False
        )
        processed['input_ids'].append(encoding['input_ids'])
        processed['attention_mask'].append(encoding['attention_mask'])
        processed['label'].append(label)
    return processed

# Training Function
def train_wsd_system():
    # Load and split dataset
    processor = BengaliWSDDataProcessor(JSON_DATA_PATH)
    dataset = processor.get_dataset().train_test_split(test_size=0.2, seed=42)

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["[TGT]", "[/TGT]"], special_tokens=True)

    # Tokenize
    tokenized_ds = dataset.map(
        lambda batch: tokenize_with_markers(batch, tokenizer),
        batched=True,
        remove_columns=['sentence', 'target_word', 'sense']
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(processor.sense_labels),
        id2label=processor.id2label,
        label2id=processor.label2id
    )
    model.config.id2label = processor.id2label
    model.config.label2id = processor.label2id
    model.resize_token_embeddings(len(tokenizer))

    # Metrics function
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Training arguments and trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=37,
            load_best_model_at_end=True,
            **TRAIN_ARGS
        ),
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    # Train the model
    print("🚀 Starting training...")
    trainer.train()

    # Save the model and tokenizer
    trainer.model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    print(f"✅ Model saved to {MODEL_OUTPUT_DIR}")

# Main Execution
if __name__ == "__main__":
    train_wsd_system()
