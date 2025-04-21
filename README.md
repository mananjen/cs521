
---

# Hinglish Code-Mixed Sarcasm Detection  
**Course Project - CS521**

## 🧠 Model  
Final model used: [l3cube-hingbert](https://aclanthology.org/2022.wildre-1.2/)  
Related references:  
- https://aclanthology.org/2020.wnut-1.2/  
- https://aclanthology.org/2021.wassa-1.21/

## 📌 Overview  
This project implements a sarcasm classification system for Hinglish (Hindi-English code-mixed) social media text using **HingBERT**, a fine-tuned transformer-based model.

### Key Tools & Libraries
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- PyTorch
- HuggingFace Datasets & Tokenizers
- Indic Transliteration tools
- Emoji & slang normalization utilities

---

## ⚙️ Installation & Setup

Install the required Python dependencies:
```bash
pip install transformers datasets sklearn ftfy emoji indic-transliteration
```

Ensure CUDA is available for accelerated training:
```python
import torch
print(torch.cuda.get_device_name(0))
```

---

## 📂 Input Format & Preprocessing

### Input CSV Requirements:
- `Transliterated_Comment`: The preprocessed Hinglish comment
- `sarcasm_label`: Label indicating `"sarcastic"` or `"non-sarcastic"` (optional for inference)

### Preprocessing Steps:
- Devanagari-to-ITRANS transliteration (if required)
- Emoji aliasing (e.g., 😂 → `face_with_tears_of_joy`)
- Normalization of slang and censored words
- General cleaning using:
  - `emoji`, `ftfy`, `contractions`, `indic_transliteration`

---

## 🏗️ Model Architecture

- **Base Model**: `bert-base-multilingual-cased`, fine-tuned on Hinglish data
- **Tokenizer**: `AutoTokenizer` from HuggingFace
- **Architecture**:
  - BERT Encoder
  - Feedforward classification head for binary output
- **Data Collator**: `DataCollatorWithPadding` for dynamic padding during batching

---

## 🏋️ Training Instructions

1. Load and preprocess the dataset
2. Tokenize the text using HuggingFace tokenizer
3. Split into train and validation sets using `train_test_split` from `sklearn`
4. Convert to `DatasetDict` format
5. Use HuggingFace `Trainer` API to train

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

---
