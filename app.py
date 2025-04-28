import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import pickle

# Download stopwords (only first time)
nltk.download('stopwords')

# Define Deep Averaging Network
class DeepAveragingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepAveragingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Text Preprocessing
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Load pre-trained word embeddings
with open('word_embeddings.pkl', 'rb') as f:
    word_embeddings = pickle.load(f)

embedding_dim = 50  # Assuming 50-dim embeddings (adjust if different)

# Load trained model
model = DeepAveragingNetwork(input_dim=embedding_dim, hidden_dim=100, output_dim=3)  # Change output_dim based on your number of classes
model.load_state_dict(torch.load('trained_dan_model.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels
class_labels = ['class1', 'class2', 'class3']  # <-- Replace with your actual class names

# Function to get averaged embedding
def get_text_embedding(tokens):
    vectors = []
    for token in tokens:
        if token in word_embeddings:
            vectors.append(word_embeddings[token])
    if not vectors:
        return torch.zeros(embedding_dim)
    avg_vector = np.mean(vectors, axis=0)
    return torch.tensor(avg_vector, dtype=torch.float32)

# Streamlit App
st.title("Text Representation & Deep Averaging Network")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.error("Please enter some text!")
    else:
        tokens = preprocess(user_input)
        text_embedding = get_text_embedding(tokens)
        output = model(text_embedding)
        prediction = torch.argmax(output).item()
        st.success(f"Prediction: {class_labels[prediction]}")
