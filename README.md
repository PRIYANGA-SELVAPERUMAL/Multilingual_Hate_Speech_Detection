# Multilingual_Hate_Speech_Detection
This project focuses on detecting hate speech, stress, and anxiety-related content in multilingual social media posts, specifically using the MPOX dataset collected from Instagram. The dataset includes 60,127 posts across 52 languages, annotated for hate speech detection, sentiment analysis, and stress/anxiety classification.

**🔍 Key Features:**
**Dataset Processing:** Text cleaning, tokenization, lemmatization, and translation handling.
🔍 Key Features:
**Dataset Processing:** Text cleaning, tokenization, lemmatization, and translation handling.

**Feature Engineering:**
Traditional ML features: TF-IDF, Word2Vec, handcrafted linguistic features
Transformer-based embeddings: WordPiece tokenization, positional embeddings

**Model Implementation:**
Machine Learning Models: Random Forest, XGBoost, LightGBM, SVM, Logistic Regression
Transformer Models: mBERT, BERT, DistilBERT, RoBERTa, XLNet

**Training & Evaluation:**
Data split: 80:10:10 (train/validation/test)
Optimized using Adam optimizer & early stopping
Evaluation metrics: Accuracy, Precision, Recall, F1-score, MSE, MAPE

**Auxiliary NLP Tasks:**
Named Entity Recognition (NER)
Sentiment Analysis
Toxicity & Offensive Language Detection

**📊 Results:**
Transformer models outperform traditional ML approaches, with mBERT achieving the highest accuracy (~97.7%).
The study highlights the effectiveness of multilingual embeddings in social media hate speech detection.

**🚀 Future Work:**
Expanding dataset coverage to more languages
Fine-tuning transformer models on domain-specific data
Exploring zero-shot learning for low-resource languages

**📌 Use Cases:**
✅ Social media monitoring
✅ Hate speech moderation
✅ Mental health analysis on social platforms

