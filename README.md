# 📧 Spam / Ham Classification Project

An end-to-end machine learning project that detects whether a message (SMS/Email text) is **SPAM** or **HAM (Not Spam)** using **Word2Vec + RandomForest Classifier** and deployed using **Streamlit**.

---

## 🚀 Project Overview

This project classifies text messages into two categories:

* **Spam** – unwanted messages often containing promotions, scams, offers
* **Ham** – normal, genuine messages sent by people

The model is trained on the popular **SMS Spam Collection Dataset** and uses:

* **NLP preprocessing** (cleaning, stopword removal, stemming)
* **Word Embeddings** using **Word2Vec**
* **Vector averaging** to get document-level features
* **RandomForest Classifier** for prediction
* **Streamlit Web App** for easy UI

---

## 🗂️ Project Folder Structure

```
SPAM_HAM_classification/
│
├── spam_app/                     # Streamlit UI Application
│     ├── app.py                  # Main Streamlit app
│     ├── spam_model.pkl          # Saved RandomForest model
│     ├── word2vec.model          # Saved Word2Vec embeddings
│     ├── stopwords.pkl           # Saved stopword list
│     ├── stemmer.pkl             # Saved PorterStemmer object
│
├── spam.csv                      # SMS spam dataset
├── requirements.txt              # Python dependencies
├── venv/                         # Virtual environment
└── training_notebooks.ipynb      # Model training notebook (optional)
```

---

## ⚙️ Technologies Used

### **📌 Libraries / Frameworks**

* Python 3.x
* NumPy, Pandas
* NLTK
* Gensim (Word2Vec)
* Scikit-learn
* Streamlit
* Pickle

---

## 🧹 NLP Preprocessing Steps

Each message goes through:

1. Removing special characters
2. Converting to lowercase
3. Tokenization (splitting into words)
4. Removing stopwords
5. Stemming using PorterStemmer
6. Word2Vec embedding
7. Averaging vectors → final 100-dim vector per message

---

## 🤖 Model Training

### **Algorithm used:**

* **RandomForestClassifier**

### **Feature Extraction:**

* **Word2Vec (100 dimensions)**
* Average embedding per message

---

## 📈 Model Performance

After training:

```
Confusion Matrix:
[[951  15]
 [ 24 125]]

Accuracy: 97%
Precision (Spam): 0.89
Recall (Spam): 0.84
F1 Score: 0.87
```

The model performs very well, with high accuracy and excellent ham detection.

---

## 🌐 Streamlit Web App

The UI is built using Streamlit for quick deployment and interaction.

### **Features:**

* Enter any text (SMS/Email)
* Click **Predict**
* See SPAM / NOT SPAM instantly
* Shows confidence score (if enabled)

### **Run the App:**

```bash
cd spam_app
streamlit run app.py
```

Your browser will open automatically.

---

## 🔧 How It Works Internally

1. User enters a message.
2. Message is preprocessed.
3. Converted into tokens.
4. Averaged Word2Vec vector created.
5. Passed into RandomForest model.
6. Prediction displayed on UI.

---

## 📦 requirements.txt

Make sure your environment contains:

```
streamlit
scikit-learn
pandas
numpy
nltk
gensim
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

* Add TF-IDF version of the model
* Add BERT/Transformer model for higher accuracy
* Deploy on Streamlit Cloud, Render, or HuggingFace Spaces
* Add HTML / CSS based Flask web app
* Add email body + subject + metadata parsing

---

## 👤 Author

**Aarav Kumar** — B.Tech CSE
This project is part of learning NLP, ML, and model deployment.

---

## 📞 Contact

If you want to extend or use this project, feel free to reach out!

---

### ⭐ Thank you for exploring this project!

Feel free to add this link in your portfolio or resume.
