
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


#for ploting(visualization)
!pip install matplotlib seaborn scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm



# Configurations
JSON_DATA_PATH = "./new-data.json"
MODEL_NAME = "csebuetnlp/banglabert"
MODEL_OUTPUT_DIR = "csebuetnlp_bangla_wsd_model"
TRAIN_ARGS = {
    "learning_rate": 3e-5,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
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

def train_wsd_system():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

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

    # Trainer setup
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

    # Save model and tokenizer
    trainer.model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"✅ Model saved to {MODEL_OUTPUT_DIR}")

    # 📊 PLOTS AND METRICS
    print("📈 Generating plots and metrics...")
    
    %matplotlib inline
    font_path = "NotoSansBengali-Regular.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = font_prop.get_name()

    # Register with Matplotlib's font manager
    fm.fontManager.addfont(font_path)
    print("Font loaded:", font_prop.get_name())

    # rcParams
    plt.rcParams['font.family'] = font_prop.get_name()
    # 1. Training and Eval Loss Plot
    loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    loss_epochs = [log['epoch'] for log in trainer.state.log_history if 'loss' in log]

    eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_loss_epochs = [log['epoch'] for log in trainer.state.log_history if 'eval_loss' in log]

    eval_acc = [log['eval_accuracy'] for log in trainer.state.log_history if 'eval_accuracy' in log]
    eval_acc_epochs = [log['epoch'] for log in trainer.state.log_history if 'eval_accuracy' in log]

    
# ✅ 1. Training and Eval Loss Plot
    if loss_values and loss_epochs and len(loss_values) == len(loss_epochs):
       plt.figure(figsize=(10, 5))
       plt.plot(loss_epochs, loss_values, label='Training Loss')
       if eval_loss and eval_loss_epochs and len(eval_loss) == len(eval_loss_epochs):
          plt.plot(eval_loss_epochs, eval_loss, label='Eval Loss', linestyle='--')
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.title('Training and Evaluation Loss')
       plt.legend()
       plt.grid()
       plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'loss_plot.png'))
       plt.close()
    else:
       print("⚠️ Insufficient or mismatched data to plot loss curves.")


# ✅ 2. Accuracy Plot
    if eval_acc and eval_acc_epochs and len(eval_acc) == len(eval_acc_epochs):
       plt.figure(figsize=(10, 5))
       plt.plot(eval_acc_epochs, eval_acc, label='Eval Accuracy', color='green')
       plt.xlabel('Epoch')
       plt.ylabel('Accuracy')
       plt.title('Evaluation Accuracy per Epoch')
       plt.legend()
       plt.grid()
       plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'accuracy_plot.png'))
       plt.close()
    else:
       print("⚠️ Insufficient or mismatched data to plot accuracy curve.")


    # 3. Classification Report Heatmap
    target_labels = [processor.id2label[i] for i in range(len(processor.id2label))]
    predictions = trainer.predict(tokenized_ds["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    report = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    zero_division=0,
    labels=list(range(len(target_labels))),
    target_names=target_labels
    )

    metrics = ['precision', 'recall', 'f1-score']
    report_matrix = np.array([[report[label][metric] for metric in metrics] for label in target_labels])

    plt.figure(figsize=(12, 6))
    sns.heatmap(report_matrix, annot=True, xticklabels=metrics, yticklabels=target_labels, cmap="YlGnBu", fmt=".2f")
    plt.title("Classification Report Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'classification_report_heatmap.png'))
    plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_labels, yticklabels=target_labels, cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

# Main Execution
if __name__ == "__main__":
    train_wsd_system()
