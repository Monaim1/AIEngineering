import os
import requests
import tiktoken
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json



torch.manual_seed(1337)
# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_file_path = os.path.join('Data', 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()


# this is just for character-level tokentization as i only have a mac M4
# Sentencepiece is whats used commonly within the NLP community
chars = sorted(list(set(text)))
vocab_size = len(chars) # gpt2 is around 50K dimensional embeddings

# Encode  decode 
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        # Get a chunk of text of size block_size + 1
        chunk = self.data[idx:idx + self.block_size + 1]
    
        # Split into input and target
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device)
        return x, y


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by number of heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Split the embedding into num_heads different pieces
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
        
        # Scaled dot-product attention
        attention = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        attention = attention / (self.embed_size ** (1/2))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        # Apply softmax
        attention = F.softmax(attention, dim=-1)
        
        out = torch.einsum("bhql,blhd->bqhd", [attention, values])
        out = out.reshape(batch_size, seq_len, self.embed_size)
        
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attention = self.attention(x, mask)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        x = self.dropout(x)
        
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        x = self.embedding(x) + self.pos_embedding(positions)
        
        # Create a causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to(x.device)

        for layer in self.layers:
            x = layer(x, mask)
            
        logits = self.fc_out(x)
        
        if targets is None:
            return logits, None
            
        B, T, C = logits.shape
        logits_flat = logits.view(B*T, C)
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop idx to the last block_size tokens
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]  # Get the last time step
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx


def save_model(model, path="transformer_model.pth"):
    """Saves the model state to a file."""
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)
    print("Model saved.")

def load_model(model, path="transformer_model.pth"):
    """Loads the model state from a file."""
    if os.path.exists(path):
        print(f"Loading model from {path}...")
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print("Model loaded.")
    else:
        print(f"No model found at {path}, training from scratch.")


################################################################################
## hyperparams:
block_size= 32
batch_size = 32
embed_size=64
num_heads=4
num_layers=4
learning_rate=3e-4


tokenized_data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = len(tokenized_data)
train_data = tokenized_data[:int(n*0.9)]
val_data = tokenized_data[int(n*0.9):]

train_dataset = CharDataset(train_data.cpu(), block_size = block_size)  # Keep data on CPU for DataLoader
val_dataset = CharDataset(val_data.cpu(), block_size = block_size) 

# Create data loaders

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Create model and move to device
# Initialize model
model = SimpleTransformer(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Training
num_epochs = 3
print("Starting Training: ")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x_batch, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, loss = model(x_batch, y_batch)
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}', 
          f'Train Loss: {train_loss/len(train_loader)}', 
          f'Val Loss: {val_loss/len(val_loader)}')

# Save the final model
save_model(model)

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=2000)
print(decode(generated[0].cpu().tolist()))  # Move back to CPU for decoding