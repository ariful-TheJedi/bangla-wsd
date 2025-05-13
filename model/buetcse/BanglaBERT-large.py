import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' 

def find_bengali_font():
    possible_paths = [
        "./fonts/NotoSansBengali-Regular.ttf",
        "/Library/Fonts/NotoSansBengali-Regular.ttf",
        "/Users/pcName/Library/Fonts/NotoSansBengali-Regular.ttf",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return FontProperties(fname=path)
    raise FileNotFoundError("Noto Sans Bengali font not found. Please install it or put it in ./fonts/")

# Load the font
bengali_font = find_bengali_font()

# Set it as default font for all plots
plt.rcParams['font.family'] = bengali_font.get_name()

# Optional: Confirm font name
print("🔤 Using font:", bengali_font.get_name())

# Configurations
JSON_DATA_PATH = "./new-data.json"
MODEL_NAME = "csebuetnlp/banglabert"
MODEL_OUTPUT_DIR = "csebuetnlp_bangla_wsd_model"
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

# Cleanup function for checkpoints
def cleanup_checkpoints(output_dir):
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    for checkpoint in checkpoint_dirs:
        shutil.rmtree(os.path.join(output_dir, checkpoint), ignore_errors=True)

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

    # Lists to store metrics
    train_metrics = {'epoch': [], 'loss': [], 'accuracy': []}
    eval_metrics = {'epoch': [], 'loss': [], 'accuracy': []}
    eval_predictions = []
    eval_labels = []

    # Metrics function
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids

        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        
        # Store predictions and labels for confusion matrix
        eval_predictions.extend(predictions.tolist())
        eval_labels.extend(labels.tolist())

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Custom Trainer to log metrics
    class CustomTrainer(Trainer):
        def log(self, logs):
            super().log(logs)
            epoch = logs.get('epoch', None)
            if 'loss' in logs:
                train_metrics['epoch'].append(epoch)
                train_metrics['loss'].append(logs['loss'])
                train_metrics['accuracy'].append(logs.get('eval_accuracy', 0))  # Approximate training accuracy if available
            if 'eval_loss' in logs:
                eval_metrics['epoch'].append(epoch)
                eval_metrics['loss'].append(logs['eval_loss'])
                eval_metrics['accuracy'].append(logs.get('eval_accuracy', 0))

    # Training arguments and trainer
    trainer = CustomTrainer(
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

    # Evaluate to get final metrics
    eval_results = trainer.evaluate()

    # # Quantize the model
    # quantized_model = torch.quantization.quantize_dynamic(
    #     trainer.model, {torch.nn.Linear}, dtype=torch.qint8
    # )

    # # Save quantized model
    # quantized_model.save_pretrained(MODEL_OUTPUT_DIR)
    # tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    # Save the model and tokenizer
    trainer.model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    # # Clean up checkpoints
    cleanup_checkpoints(MODEL_OUTPUT_DIR)
    
    # Generate classification report
    cls_report = classification_report(
        eval_labels,
        eval_predictions,
        target_names=processor.sense_labels,
        zero_division=0,
        output_dict=True
    )

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(eval_metrics['epoch'], eval_metrics['accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(prop=bengali_font)
    plt.savefig('./csebuetnlp/csebuetnlp_accuracy_plot.png')
    plt.close()

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['epoch'], train_metrics['loss'], label='Training Loss', marker='o')
    plt.plot(eval_metrics['epoch'], eval_metrics['loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(prop=bengali_font)
    plt.savefig('./csebuetnlp/csebuetnlp_loss_plot.png')
    plt.close()
    
    # Filter out 'accuracy', 'macro avg', etc.
    # Convert to DataFrame (excluding "accuracy", "macro avg", "weighted avg")
    # Create DataFrame from classification report dict
    metrics_df = pd.DataFrame(cls_report).T.drop(["accuracy", "macro avg", "weighted avg"])
    metrics_df = metrics_df[['precision', 'recall', 'f1-score']]
    
    # Top 8
    top_8_df = metrics_df.sort_values(by="f1-score", ascending=False).head(8)
    all_labels = top_8_df.index.tolist()
    print(all_labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_8_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Per-Class Precision, Recall, and F1-score", fontproperties=bengali_font)
    plt.xlabel("Metric")
    plt.ylabel("Class Label", fontproperties=bengali_font)
    plt.xticks(fontsize=12)
    plt.yticks(rotation=0, fontproperties=bengali_font, fontsize=10)
    plt.tight_layout()
    plt.savefig("./csebuetnlp/classification_report_heatmap_top_8.png")
    plt.close()

    # ------------------ Top 8 Classes CM ------------------
    cm = confusion_matrix(eval_labels, eval_predictions)
    labels = processor.sense_labels
    
    # Get top 8 class names in the correct order
    top_8_class_names = top_8_df.index.tolist()
    
    # Map class indices
    label_to_index = {label: i for i, label in enumerate(labels)}
    top_8_indices = [label_to_index[label] for label in top_8_class_names]
    
    # Slice confusion matrix
    top_8_cm = cm[np.ix_(top_8_indices, top_8_indices)]
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_8_cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_8_class_names, yticklabels=top_8_class_names)
    plt.xlabel("Predicted", fontproperties=bengali_font)
    plt.ylabel("True Label", fontproperties=bengali_font)
    plt.title("Confusion Matrix of Top 8 F1-Score Classes", fontproperties=bengali_font)
    plt.xticks(rotation=45, ha='right', fontproperties=bengali_font)
    plt.yticks(rotation=0, fontproperties=bengali_font)
    plt.tight_layout()
    plt.savefig("./csebuetnlp/confusion_matrix_top8.png")
    plt.close()

    return trainer, eval_metrics, train_metrics, cls_report


# Predictor Class
class BengaliWSDPredictor:
    def __init__(self, model_dir, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.id2label = self.model.config.id2label  # Load from config
        self.device = device

    def predict(self, sentences, target_words):
        if isinstance(sentences, str):
            sentences = [sentences]
            target_words = [target_words]
        
        marked_sents = [sent.replace(word, f"[TGT]{word}[/TGT]") for sent, word in zip(sentences, target_words)]
        inputs = self.tokenizer(
            marked_sents,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_label_ids = torch.argmax(outputs.logits, dim=1).tolist()
        
        return [self.id2label[pred_id] for pred_id in pred_label_ids]
        
# Main Execution
if __name__ == "__main__":
    # Train and get metrics
    trainer, eval_metrics, train_metrics, cls_report = train_wsd_system()

    # Print classification report
    print("\n📊 Classification Report")
    print(cls_report)

    predictor = BengaliWSDPredictor(MODEL_OUTPUT_DIR, device="cpu")

    test_cases = [
        ("বাজারে মাংসের দাম এত চড়া যে অনেকে নিরামিষ খাবারের দিকে ঝুঁকছে", "চড়া"),
        ("সন্ধ্যার আরতির সঙ্গে মিলিয়ে মন্দির প্রাঙ্গণে তরঙ্গের ধ্বনি ছড়িয়ে পড়ে", "তরঙ্গ"),
        ("গ্রামে পহেলা বৈশাখের পাল ছিল খুবই রঙিন এবং আনন্দময়", "পাল"),
        ("যত বেশি ধর্মীয় ধারায় মনোনিবেশ করা যায়, তত বেশি আত্মবিশ্বাসের অনুভূতি তৈরি হয়", "ধারা")
    ]
    print("\n🔍 Test Predictions")
    for sentence, word in test_cases:
        prediction = predictor.predict(sentence, word)
        print(f"Sentence: {sentence}")
        print(f"Target Word: '{word}' → Predicted Sense: {prediction}")
        print("-" * 60)

