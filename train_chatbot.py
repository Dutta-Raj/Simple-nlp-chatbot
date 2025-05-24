import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

def load_intents(file_path="intents.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def prepare_data(data):
    X = []
    y = []
    for intent in data.get("intents", []):
        for pattern in intent.get("patterns", []):
            X.append(pattern)
            y.append(intent["tag"])
    if not X:
        raise ValueError("No training data found.")
    return X, y

def train_model(X, y):
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vec, y)
    return model, vectorizer

def save_model(model, vectorizer):
    with open("model.pkl", "wb") as mf, open("vectorizer.pkl", "wb") as vf:
        pickle.dump(model, mf)
        pickle.dump(vectorizer, vf)

def main():
    try:
        data = load_intents()
        X, y = prepare_data(data)
        model, vectorizer = train_model(X, y)
        save_model(model, vectorizer)
        print("✅ Model trained and saved successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")

main()
