from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os

class BengaliWSD_Predictor:
    def __init__(self, model_dir):
        # Convert to absolute path and normalize
        model_dir = os.path.abspath(model_dir)
        
        # Verify the path exists
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")
            
        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        
        # Load label map
        label_map_path = os.path.join(model_dir, "label_map.json")
        with open(label_map_path, encoding='utf-8') as f:
            label_map = json.load(f)
        self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
    
    def predict(self, sentence, target_word):
        marked_sent = sentence.replace(target_word, f"[TGT]{target_word}[/TGT]")
        inputs = self.tokenizer(
            marked_sent,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return self.id2label[torch.argmax(outputs.logits).item()]

# How to use:
if __name__ == "__main__":
    # For Colab:
    # predictor = BengaliWSD_Predictor("/content/drive/MyDrive/bangla_wsd_model")
    
    # For local (Linux/Mac):
    predictor = BengaliWSD_Predictor("./bangla_wsd_model")
    
    # For local (Windows):
    # predictor = BengaliWSD_Predictor(r"C:\path\to\bangla_wsd_model")
    
test_cases  = [
  ("এই চালটি করলে আমি যে পরিস্থিতি সৃষ্টি করতে পারব, সেটা সে বুঝতে পারছে না।", "চাল"),
  ("আমাদের দলের নতুন চাল ছিল প্রতিপক্ষের আক্রমণ ঠেকানো।", "চাল"), 
  ("নদীর জল খুব পরিষ্কার এবং সচ্ছ ছিল।", "জল"),
  ("বৃষ্টির দিনে একদম ভিজে গিয়েছিল, পুরো শরীর ছিল জলজল।", "জল")
]
    
for sent, word in test_cases:
        print(f"Sentence: {sent}")
        print(f"Target: '{word}' → {predictor.predict(sent, word)}")
        print("-" * 60)