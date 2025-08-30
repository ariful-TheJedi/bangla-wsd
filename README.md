# 📖 “Bengali Word Sense Disambiguation Using Supervised Approach (BERT)”

WSD is the task of determining the correct meaning of a polysemous word based on its **context**.  

Bengali, spoken by over **250 million people**, is morphologically rich and highly context-dependent. This makes WSD a particularly challenging but essential problem for applications like **machine translation, chatbots, search engines, and information retrieval**.  

---

## 🚀 Project Highlights
- **Curated Dataset**
  - 122 polysemous Bengali words  
  - 440 distinct senses  
  - ~22,600 manually annotated sentences (JSON format)

- **Models Used**
  - `csebuetnlp/BanglaBERT`
  - `sagorsarker/bangla-bert-base`

- **Pipeline**
  - Preprocessing with normalization, tokenization, and special markers `[TGT] ... [/TGT]` to highlight target words
  - Stratified dataset split: 80% training, 10% validation, 10% testing
  - Fine-tuned transformer-based BERT models for classification
  - Evaluation using **Accuracy, Precision, Recall, F1-score, Confusion Matrices**

- **Performance**
  - `BanglaBERT`: **88.60% accuracy**  
    - Precision: 88.93% | Recall: 89.82% | F1-score: 89.38%  
  - `sagorsarker/bangla-bert-base`: **83.12% accuracy**  
  - BanglaBERT achieved stronger generalization across distinct categories, with some challenges in nuanced/low-resource senses  

---

## 📊 Example Input & Output

### Example 1
**Input Sentence:**  
সে [TGT]চাল[/TGT] বাজার থেকে কিনে এনেছে।
**Model Prediction:**  
- Correct Sense: *Rice (food grain)* ✅  

---

### Example 2
**Input Sentence:**  
সে খুব দ্রুত গাড়ি [TGT]চাল[/TGT]াচ্ছে।
**Model Prediction:**  
- Correct Sense: *Driving (to operate a vehicle)* ✅  

---

