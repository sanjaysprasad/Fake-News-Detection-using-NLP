import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import joblib

print("Starting training script...")

# Download NLTK stopwords (comment out after first run)
nltk.download('stopwords')

# Set your dataset folder path here
folder_path = 'C:/Users/Sanjay New/Documents/SSP CODES/Fake news NLP/Dataset'

print("Loading datasets...")   
file_fake = os.path.join(folder_path, 'Fake.csv')
file_true = os.path.join(folder_path, 'True.csv')

df_fake = pd.read_csv(file_fake)
df_fake['label'] = 0  # Fake news label

df_true = pd.read_csv(file_true)
df_true['label'] = 1  # Real news label

df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset size: {len(df)}")
print("Sample data:")
print(df.head())

if 'text' not in df.columns:
    raise ValueError("Dataset must have a 'text' column.")

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("Preprocessing text...")
df['clean_text'] = df['text'].apply(preprocess)

print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Passive Aggressive Classifier...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(cm)

# Save the model and vectorizer for later use
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved to 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'")

print("Training script finished.")
