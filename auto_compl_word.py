import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import re
import matplotlib.pyplot as plt

# Load and preprocess dataset
file_path = "autocompletion/dataset/data.txt"  # Ensure dataset is available before running

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text.split()

# Load and preprocess entire dataset
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
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)  # Flatten embeddings
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model
model = MLPWordPredictor(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with loss tracking after each backpropagation
epochs = 50
losses = []  # To store loss for visualization
for epoch in range(epochs):
    for i in range(len(X_train)):  # Iterate over each sample in the dataset (or use batches for larger datasets)
        optimizer.zero_grad()
        
        # Get the context and target from X_train and Y_train
        context_idx = X_train[i].unsqueeze(0)  # Add batch dimension
        target_idx = Y_train[i].unsqueeze(0)   # Add batch dimension
        
        # Forward pass
        logits = model(context_idx)
        
        # Compute loss
        loss = criterion(logits, target_idx)
        
        # Backward pass
        loss.backward()
        
        # Update the model weights
        optimizer.step()
        
        # Track loss after backpropagation
        losses.append(loss.item())
        
        # Print loss after each update (optional)
        if (i+1) % 100 == 0:  # Print loss every 100 updates
            print(f"Epoch {epoch+1}/{epochs}, Sample {i+1}/{len(X_train)}, Loss: {loss.item():.4f}")

# Plotting the loss curve
plt.plot(range(len(losses)), losses, label='Training Loss')
plt.xlabel('Steps (backpropagation)')
plt.ylabel('Loss')
plt.title('Training Loss After Each Backpropagation')
plt.legend()
plt.show()



# Plotting the loss curve
plt.plot(range(epochs), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()


# Prediction Function
def predict_next_word(model, context_words):
    model.eval()
    context_idx = torch.tensor([[word_to_index[word] for word in context_words]], dtype=torch.long)
    with torch.no_grad():
        logits = model(context_idx)
        predicted_idx = torch.argmax(logits, dim=1).item()
    return index_to_word[predicted_idx]

# Example Prediction
context_example = ["why", "are"]  # Replace with words from the dataset
if all(word in word_to_index for word in context_example):
    predicted_word = predict_next_word(model, context_example)
    print(f"Predicted next word: {predicted_word}")
