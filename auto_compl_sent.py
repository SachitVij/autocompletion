import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import re

# Load and preprocess dataset
file_path = "autocompletion/dataset/data.txt"

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text.split()

# Load and preprocess dataset
chat_data = read_file(file_path)
tokenized_sentences = [preprocess_text(line) for line in chat_data]
tokenized_words = [word for sentence in tokenized_sentences for word in sentence]

# Build Vocabulary
vocab_size = 5000
word_counts = Counter(tokenized_words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)[:vocab_size]
word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Create Training Data
context_size = 2
training_data = []
for sentence in tokenized_sentences:
    for i in range(len(sentence) - context_size):
        context = sentence[i:i + context_size]
        target = sentence[i + context_size]
        if target in word_to_index and all(word in word_to_index for word in context):
            training_data.append(([word_to_index[word] for word in context], word_to_index[target]))

# Convert to PyTorch tensors
X_train = torch.tensor([x for x, y in training_data], dtype=torch.long)
Y_train = torch.tensor([y for x, y in training_data], dtype=torch.long)

# Define MLP Model
class MLPWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)  # Flatten embeddings
        x = torch.relu(self.fc1(x))  # Activation
        x = self.fc2(x)
        return x

# Initialize Model
model = MLPWordPredictor(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 30
for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, Y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Sentence Generation Function
def generate_sentence(model, start_words, max_length=10):
    model.eval()
    sentence = start_words[:]
    for _ in range(max_length):
        context_idx = torch.tensor([[word_to_index[word] for word in sentence[-context_size:]]], dtype=torch.long)
        with torch.no_grad():
            logits = model(context_idx)
            predicted_idx = torch.argmax(logits, dim=1).item()
        sentence.append(index_to_word[predicted_idx])
        if sentence[-1] == "end":  # Stop if the model predicts an end token
            break
    return " ".join(sentence)

# Example Sentence Generation
start_words = ["i", "am"]
if all(word in word_to_index for word in start_words):
    generated_sentence = generate_sentence(model, start_words)
    print(f"Generated Sentence: {generated_sentence}")
