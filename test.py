import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if needed; comment out after first run
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def test_bench():
    print("Fake News Detection Test Bench")
    print("Enter news text to classify it as Fake or Real News.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter news text:\n")
        if user_input.lower() == "exit":
            print("Exiting Test Bench.")
            break

        cleaned_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        label = "Real News" if prediction == 1 else "Fake News"
        print(f"Prediction: {label}\n")

if __name__ == "__main__":
    test_bench()
