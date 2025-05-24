import pickle
import numpy as np

# Load saved model and vectorizer
with open("model.pkl", "rb") as mf:
    model = pickle.load(mf)

with open("vectorizer.pkl", "rb") as vf:
    vectorizer = pickle.load(vf)

# Define possible responses (you can extend this or load from intents.json)
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    "goodbye": ["Goodbye!", "Take care!", "See you soon!"],
    "thanks": ["You're welcome!", "No problem!", "Any time!"],
    "name": ["I'm your AI assistant!", "I'm a chatbot created with Python."]
}


def chatbot_response(text):
    # Vectorize the input text
    text_vec = vectorizer.transform([text])
    # Predict intent tag
    pred = model.predict(text_vec)[0]
    # Pick a random response for that intent
    return np.random.choice(responses.get(pred, ["Sorry, I don't understand."]))

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    reply = chatbot_response(user_input)
    print("Chatbot:", reply)
