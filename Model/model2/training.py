'''
Model Name: sagorsarker/bangla-bert-base
Author Detail: https://huggingface.co/sagorsarker/bangla-bert-base


This code is coolab ready code
You can run this code on pc with some changes(dataset directory, save model directory, font directory etc...)
Change dataset directory
The model size can be huge so be coutious about that
Try to save the model inside the drive
'''

import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
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
import shutil

def find_bengali_font():
    # possible_paths = [
    #     "./Fonts/NotoSansBengali-Regular.ttf",
    #     "C:/Windows/Fonts/NotoSansBengali-Regular.ttf",  # Default Windows font directory
    #     os.path.join(os.getenv("USERPROFILE"), "AppData", "Local", "Microsoft", "Windows", "Fonts", "NotoSansBengali-Regular.ttf")  # User-installed fonts
    # ]

    possible_paths = [
        "./fonts/NotoSansBengali-Regular.ttf",   # Local path (upload to Colab if needed)
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf"  # Linux/Colab default path
    ]

    # Check existing paths
    for path in possible_paths:
        if os.path.exists(path):
            return FontProperties(fname=path)

    # If not found, download from Google Noto Fonts
    print("⚠️ Bengali font not found. Downloading...")
    os.makedirs("./fonts", exist_ok=True)
    os.system("wget -q https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf -O ./fonts/NotoSansBengali-Regular.ttf")

    if os.path.exists("./fonts/NotoSansBengali-Regular.ttf"):
        return FontProperties(fname="./fonts/NotoSansBengali-Regular.ttf")
    else:
        print("⚠️ Could not load Bengali font, using default matplotlib font.")
        return FontProperties()  # fallback default


# Load the font
bengali_font = find_bengali_font()

# Set it as default font for all plots
plt.rcParams['font.family'] = bengali_font.get_name()

# Optional: Confirm font name
print("🔤 Using font:", bengali_font.get_name())

# Configurations
JSON_DATA_PATH = "/content/drive/MyDrive/Capstone/new-data.json"
MODEL_NAME = "sagorsarker/bangla-bert-base"
MODEL_OUTPUT_DIR = "/content/drive/MyDrive/Capstone/new_sagorsarkar_bangla_wsd_model"
TRAIN_ARGS = {
    "learning_rate": 3e-5,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
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
    train_metrics = {'epoch': [], 'loss': [], 'accuracy': []}
    eval_metrics = {'epoch': [], 'loss': [], 'accuracy': []}
    eval_predictions, eval_labels = [], []


    # Load and split dataset
    processor = BengaliWSDDataProcessor(JSON_DATA_PATH)
    full_dataset = processor.get_dataset()

    # Convert HF Dataset to pandas DataFrame
    df = full_dataset.to_pandas()

    # Step 1: Train (80%) and temp (20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],  # stratify by class labels
        random_state=42
    )

    # Step 2: Validation (10%) and Test (10%) from temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )

    # Convert back to HF Dataset
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["[TGT]", "[/TGT]"], special_tokens=True)

    # Tokenize
    # tokenized_ds = dataset.map(
    #     lambda batch: tokenize_with_markers(batch, tokenizer),
    #     batched=True,
    #     remove_columns=['sentence', 'target_word', 'sense']
    # )

    #tokenize training dataset
    tokenize_train_dataset = train_dataset.map(
        lambda batch: tokenize_with_markers(batch, tokenizer),
        batched=True,
        remove_columns=['sentence', 'target_word', 'sense']
    )



    #tokenize validation dataset
    tokenize_validation_dataset = validation_dataset.map(
        lambda batch: tokenize_with_markers(batch, tokenizer),
        batched=True,
        remove_columns=['sentence', 'target_word', 'sense']
    )

    #tokenize testing dataset
    tokenize_test_dataset = test_dataset.map(
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
    # def compute_metrics(eval_pred):
    #     predictions = np.argmax(eval_pred.predictions, axis=1)
    #     labels = eval_pred.label_ids

    #     acc = accuracy_score(labels, predictions)
    #     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)

    #     # Store predictions and labels for confusion matrix
    #     eval_predictions.extend(predictions.tolist())
    #     eval_labels.extend(labels.tolist())

    #     # return {
    #     #     'accuracy': acc,
    #     #     'precision': precision,
    #     #     'recall': recall,
    #     #     'f1': f1
    #     # }

    def compute_metrics(eval_pred):
           predictions = np.argmax(eval_pred.predictions, axis=1)
           labels = eval_pred.label_ids

           acc = accuracy_score(labels, predictions)
           precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
               )

           # store for later analysis
           eval_predictions.extend(predictions.tolist())
           eval_labels.extend(labels.tolist())

           return {
             "accuracy": acc,
             "precision": precision,
             "recall": recall,
             "f1": f1
          }


    # Custom Trainer to log metrics
    class CustomTrainer(Trainer):
        def log(self, logs, *args, **kwargs):
            super().log(logs, *args, **kwargs)
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
        # train_dataset=tokenized_ds["train"],
        # eval_dataset=tokenized_ds["test"],
        train_dataset=tokenize_train_dataset,
        eval_dataset = tokenize_validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    # Train the model
    print("🚀 Starting training...")
    trainer.train()

    # Evaluate to get final metrics
    eval_results = trainer.evaluate()

    # Quantize the model
    # quantized_model = torch.quantization.quantize_dynamic(
    #     trainer.model, {torch.nn.Linear}, dtype=torch.qint8
    # )

    test_results = trainer.evaluate(eval_dataset=tokenize_test_dataset)
    print("🧪 Test Dataset Results:", test_results)

    # Save quantized model
    # quantized_model.save_pretrained(MODEL_OUTPUT_DIR)
    # tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    # Save the model and tokenizer
    trainer.model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    #Clean up checkpoints
    cleanup_checkpoints(MODEL_OUTPUT_DIR)

    # get only the labels present in eval_labels
    used_labels = unique_labels(eval_labels, eval_predictions)
    used_target_names = [processor.id2label[i] for i in used_labels]

     # Generate classification report
    cls_report = classification_report(
    eval_labels,
    eval_predictions,
    labels=used_labels,
    target_names=used_target_names,
    zero_division=0,
    output_dict=True
    )

    # Generate classification report
    # cls_report = classification_report(
    #     eval_labels,
    #     eval_predictions,
    #     target_names=processor.sense_labels,
    #     zero_division=0,
    #     output_dict=True
    # )

    # Plot Training and Validation Accuracy
    #plt.figure(figsize=(10, 6))
    # plt.plot(eval_metrics['epoch'], eval_metrics['accuracy'], label='Validation Accuracy', marker='o')
    # plt.title('Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.legend(prop=bengali_font)
    # plt.savefig('./csebuetnlp/csebuetnlp_accuracy_plot.png')
    # plt.close()

    # Plot Training and Validation Loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_metrics['epoch'], train_metrics['loss'], label='Training Loss', marker='o')
    # plt.plot(eval_metrics['epoch'], eval_metrics['loss'], label='Validation Loss', marker='o')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.legend(prop=bengali_font)
    # plt.savefig('./csebuetnlp/csebuetnlp_loss_plot.png')
    # plt.close()

    # Filter out 'accuracy', 'macro avg', etc.
    # Convert to DataFrame (excluding "accuracy", "macro avg", "weighted avg")
    # Create DataFrame from classification report dict
    # target_labels = ['আঘাত বা চপেটাঘাত', 'তীব্র বা উগ্র', 'উচ্চ মূল্য বা দাম', 'চোখের রোগ',
    #              'ভ্যাকসিন', 'পশুর পাল', 'কোনো অনুষ্ঠান বা উৎসবের পাল', 'কপালে পরা অলংকার/সাজসজ্জা']
    # metrics_df = pd.DataFrame(cls_report).T.drop(["accuracy", "macro avg", "weighted avg"])
    # metrics_df = metrics_df[['precision', 'recall', 'f1-score']]

    # Top 8
    # top_8_df = metrics_df.sort_values(by="f1-score", ascending=False).head(8)
    # all_labels = top_8_df.index.tolist()
    # selected_df = metrics_df.loc[target_labels]
    # # print(all_labels)
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(selected_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    # plt.title("Per-Class Precision, Recall, and F1-score", fontproperties=bengali_font)
    # plt.xlabel("Metric")
    # plt.ylabel("Class Label", fontproperties=bengali_font)
    # plt.xticks(fontsize=12)
    # plt.yticks(rotation=0, fontproperties=bengali_font, fontsize=10)
    # plt.tight_layout()
    # plt.savefig("./csebuetnlp/classification_report_heatmap_top_8.png")
    # plt.close()

    # ------------------ Top 8 Classes CM ------------------
    # labels = processor.sense_labels  # full label list used for classification
    # cm = confusion_matrix(eval_labels, eval_predictions)

    # # Map label to index
    # label_to_index = {label: idx for idx, label in enumerate(labels)}

    ##Get indices for your specific labels
    #selected_indices = [label_to_index[label] for label in target_labels]

    ##Slice the confusion matrix
    #selected_cm = cm[np.ix_(selected_indices, selected_indices)]


    # plt.figure(figsize=(10, 8))
    # sns.heatmap(selected_cm, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=target_labels, yticklabels=target_labels)
    # plt.xlabel("Predicted", fontproperties=bengali_font)
    # plt.ylabel("True Label", fontproperties=bengali_font)
    # plt.title("Confusion Matrix for Selected Classes", fontproperties=bengali_font)
    # plt.xticks(rotation=45, ha='right', fontproperties=bengali_font)
    # plt.yticks(rotation=0, fontproperties=bengali_font)
    # plt.tight_layout()
    # plt.savefig("./csebuetnlp/confusion_matrix_top8.png")
    # plt.close()

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



    #prediction part(Testing)
    #predictor = BengaliWSDPredictor(MODEL_OUTPUT_DIR, device="cpu")

    # test_cases = [
    #     ("বাজারে মাংসের দাম এত চড়া যে অনেকে নিরামিষ খাবারের দিকে ঝুঁকছে", "চড়া"),
    #     ("সন্ধ্যার আরতির সঙ্গে মিলিয়ে মন্দির প্রাঙ্গণে তরঙ্গের ধ্বনি ছড়িয়ে পড়ে", "তরঙ্গ"),
    #     ("গ্রামে পহেলা বৈশাখের পাল ছিল খুবই রঙিন এবং আনন্দময়", "পাল"),
    #     ("যত বেশি ধর্মীয় ধারায় মনোনিবেশ করা যায়, তত বেশি আত্মবিশ্বাসের অনুভূতি তৈরি হয়", "ধারা")
    # ]
    # print("\n🔍 Test Predictions")
    # for sentence, word in test_cases:
    #     prediction = predictor.predict(sentence, word)
    #     print(f"Sentence: {sentence}")
    #     print(f"Target Word: '{word}' → Predicted Sense: {prediction}")
    #     print("-" * 60)
