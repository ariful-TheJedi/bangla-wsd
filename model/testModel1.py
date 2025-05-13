# 1. Install required packages (without wandb)
!pip install -q transformers datasets torch scikit-learn

# 2. Import libraries
import json
import os
import torch
import numpy as np
from sklearn.metrics import classification_report
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# 3. Mount Google Drive (if your data is stored there)
#from google.colab import drive
#drive.mount('/content/drive')

# 4. Configuration (modify these paths as needed)
JSON_DATA_PATH = "bangla_wsd_data.json"  # Change to your path
MODEL_NAME = "sagorsarker/bangla-bert-base"
MODEL_OUTPUT_DIR = "bangla_wsd_model"  # Where to save model
TRAIN_ARGS = {
    "learning_rate": 3e-5,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "report_to": "none"  # Disables wandb explicitly
}

# 5. Data Processor Class
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

# 6. Tokenization Function
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

# 7. Training Function
def train_wsd_system():
    # Initialize data
    processor = BengaliWSDDataProcessor(JSON_DATA_PATH)
    dataset = processor.get_dataset().train_test_split(test_size=0.2, seed=42)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["[TGT]", "[/TGT]"], special_tokens=True)
    
    # Tokenize datasets
    tokenized_ds = dataset.map(
        lambda batch: tokenize_with_markers(batch, tokenizer),
        batched=True,
        remove_columns=['sentence', 'target_word', 'sense']
    )
    
    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(processor.sense_labels),
        id2label=processor.id2label,
        label2id=processor.label2id
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            load_best_model_at_end=True,
            **TRAIN_ARGS
        ),
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: classification_report(
            p.label_ids, np.argmax(p.predictions, axis=1),
            target_names=processor.sense_labels,
            output_dict=True,
            zero_division=0
        )
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save everything
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    with open(f"{MODEL_OUTPUT_DIR}/label_map.json", "w") as f:
        json.dump({
            "id2label": processor.id2label,
            "label2id": processor.label2id
        }, f, ensure_ascii=False)
    
    return trainer

# 8. Main Execution
if __name__ == "__main__":
    # Train the model
    trainer = train_wsd_system()
    
    # Load predictor
    predictor = BengaliWSDPredictor(MODEL_OUTPUT_DIR)
    
    # Test predictions
    test_cases = [
        ("চালের দাম বেড়ে যাওয়ায় আমরা কম কিনেছি", "চাল"),
        ("তার চাল-চলন খুব শান্ত", "চাল"),
        ("নদীর জল আজকাল খুব দূষিত", "জল"),
        ("তার চোখে জল দেখে আমি আবেগাপ্লুত হলাম", "জল")
    ]
    
    print("\nTest Predictions:")
    for sentence, word in test_cases:
        pred = predictor.predict(sentence, word)
        print(f"Sentence: {sentence}")
        print(f"Target: '{word}' → Predicted sense: {pred}")
        print("-" * 60)