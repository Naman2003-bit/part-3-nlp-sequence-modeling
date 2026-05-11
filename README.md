# Part 3: NLP and Sequence Modeling Mini Project

## Objective
Build an end-to-end NLP pipeline to automatically classify customer
support tickets by sentiment using traditional ML and deep sequence models.

## Dataset Overview
- File   : customer_support_text_classification.csv
- Target : sentiment_label (negative, neutral, positive)
- Channels: chat, phone, email
- Key columns: customer_message, word_count, urgent_flag

## Text Preprocessing Applied
- Converted all text to lowercase
- Removed digits and numeric characters
- Stripped all punctuation marks
- Collapsed multiple whitespace into single space
- No stopword removal to preserve full message context

## Vectorization Methods Used
1. Bag of Words (4000 features)
   Counts raw word frequencies. Simple but ignores word importance.

2. TF-IDF (4000 features, unigrams + bigrams)
   Weights words by frequency and uniqueness across documents.
   Rare meaningful words score higher than common filler words.

Text must be vectorized because all ML and deep learning models
operate on numbers, not raw string characters.

## Baseline Models Compared
| Model          | Vectorizer | Test Accuracy |
|----------------|------------|---------------|
| Naive Bayes    | TF-IDF     | ~%            |
| Linear SVM     | TF-IDF     | ~%            |

## Sequence Model — Bidirectional GRU
- Architecture : Embedding(8000,32) → BiGRU(64) → Dropout(0.4) → Dense
- Padding      : Pre-padding to length 40 tokens
- Optimizer    : Adam (lr=0.0008)
- Epochs       : 12 | Batch Size : 64

## Reflection — RNNs, LSTMs, Attention, Transformers
- RNNs lose early context due to vanishing gradients over long sequences
- LSTMs fix this with input/forget/output gates and a separate cell state
- GRUs simplify LSTM with two gates achieving similar performance
- Attention lets models focus on all tokens simultaneously
- Transformers use multi-head self-attention with no recurrence,
  making them parallelizable — the foundation of GPT, BERT, LLaMA

## Repository Structure
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── eda_plots.png
    ├── model_evaluation.png
    ├── lstm_curves.png
    ├── sample_predictions.txt
    └── reflection_notes.txt
