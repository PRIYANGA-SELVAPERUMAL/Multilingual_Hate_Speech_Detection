
# 🗣️ Multilingual Hate Speech Detection using mBERT–BiLSTM

An end-to-end deep learning project that leverages a hybrid **mBERT–BiLSTM** architecture to detect hate speech in **Bengali** and **Indonesian** social media content. The system supports multilingual input, fine-tuned transformer models, and offers real-time classification — ideal for moderation and abuse monitoring platforms.

---

## 📌 Table of Contents

* [About the Project](#about-the-project)
* [Datasets](#datasets)
* [Problem Statement](#problem-statement)
* [Methodology](#methodology)
* [Model Architecture](#model-architecture)
* [Tech Stack](#tech-stack)
* [How to Run the Project](#how-to-run-the-project)
* [Results](#results)
* [Future Work](#future-work)

---

<a name="about-the-project"></a>
## 📖 About the Project

The surge in multilingual hate speech on social media platforms demands intelligent systems capable of handling code-mixed and low-resource language data. This project proposes a **hybrid mBERT–BiLSTM model** fine-tuned on Bengali and Indonesian datasets to classify text as hateful or non-hateful. It ensures real-time inference, robust evaluation, and scalable deployment potential.

---

<a name="datasets"></a>
## 📊 Datasets

### Bengali Hate Speech Dataset (Kaggle)

* \~30,000 annotated comments from Facebook & YouTube
* 2 classes:

  * `1` → Hate Speech
  * `0` → Non-Hate
* Challenges: Slang, transliteration, code-mixing
  🔗 [Bengali Dataset Link](https://www.kaggle.com/datasets/naurosromim/bengali-hate-speech-dataset)

### Indonesian Hate Speech Superset (Hugging Face)

* 14,306 Twitter posts from merged datasets
* Columns: `text`, `label`, `source`, `annotators`
* Balanced binary labels
  🔗 [Indonesian Dataset Link](https://huggingface.co/datasets/manueltonneau/indonesian-hate-speech-superset)

---

<a name="problem-statement"></a>
## ❓ Problem Statement

Design a robust multilingual hate speech detection model capable of identifying harmful content across **low-resource languages**. The system should generalize across linguistic variations and be suitable for **content moderation tools**.

---

<a name="methodology"></a>
## 🔍 Methodology

### Preprocessing:

* Cleaning: Lowercase, remove URLs/usernames, punctuation
* Tokenization: `bert-base-multilingual-cased`
* Padding/truncation to 128 tokens
* Stratified 80:20 train-test split

### Training:

* Binary classification
* Model: mBERT + 2-layer BiLSTM (128 hidden units)
* Optimizer: AdamW (LR = 2e-5)
* Loss: Weighted CrossEntropy
* Epochs: 5
* Batch Size: 16

### Inference:

* Real-time text prediction
* Text → Preprocess → Tokenize → Predict → Label

---

<a name="model-architecture"></a>
## 🧠 Model Architecture: mBERT–BiLSTM

| **Component**       | **Description**                                                |
| ------------------- | -------------------------------------------------------------- |
| **Base Model**      | `bert-base-multilingual-cased` (fine-tuned)                    |
| **Embedding Size**  | 768                                                            |
| **BiLSTM**          | 2 layers × 128 hidden units (→ 256 after bidirectional concat) |
| **Dropout**         | 0.3 (applied after BiLSTM)                                     |
| **Classifier Head** | Dense layer (Linear 256 → 2 classes)                           |
| **Loss Function**   | Weighted `CrossEntropyLoss`                                    |
| **Optimizer**       | `AdamW` with learning rate of `2e-5`                           |
| **Batch Size**      | 16                                                             |
| **Epochs**          | 5                                                              |

✅ F1-Score (Bengali): **85.67%**

✅ F1-Score (Indonesian): **82.08%**

---

<a name="tech-stack"></a>
## 🛠️ Tech Stack

| Layer         | Tools/Frameworks                   |
| ------------- | ---------------------------------- |
| Model         | PyTorch, Hugging Face Transformers |
| Tokenization  | WordPiece, Attention Masks         |
| Data Handling | Pandas, NumPy                      |
| Evaluation    | Scikit-learn, Matplotlib, Seaborn  |

---

<a name="how-to-run-the-project"></a>
## ⚙️ How to Run the Project

### 🔁 Clone the repo:

```bash
git clone https://github.com/PRIYANGA-SELVAPERUMAL/Multilingual_Hate_Speech_Detection.git
```

### 🐍 Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 📦 Install dependencies:

```bash
pip install -r requirements.txt
```

### 🚀 Run notebook or inference script:

Launch `.ipynb` files for Bengali/Indonesian or run `main.py` for terminal-based inference.

---

<a name="results"></a>
## ✅ Results

| Metric    | Bengali | Indonesian |
| --------- | ------- | ---------- |
| Accuracy  | 89.95%  | 83.61%     |
| Precision | 81.62%  | 76.33%     |
| Recall    | 90.15%  | 88.76%     |
| F1-Score  | 85.67%  | 82.08%     |
| AUC–ROC   | 0.9595  | 0.9085     |

* 📊 Confusion Matrices and AUC plots available in the results folder
* 🔍 Sample predictions included for real-world evaluation

---

<a name="future-enhancements"></a>
## 🔭 Future Enhancements

*  Add attention mechanism over BiLSTM outputs
*  Integrate XAI for explainable predictions
*  API-based deployment (Flask / FastAPI)
*  Mobile/web interface using Streamlit or React
*  Extend support to more languages (Tamil, Hindi, etc.)
*  Experiment with newer models like XLM-R and mT5



