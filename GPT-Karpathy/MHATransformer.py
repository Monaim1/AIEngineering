import os
import requests
import tiktoken
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



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
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
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
        
    def forward(self, x):
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
        
    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        x = self.dropout(x)
        
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(1000, embed_size)  # Max sequence length of 1000
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        x = self.embedding(x) + self.pos_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
            
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
                idx_cond = idx[:, -1000:]  # Limit to max sequence length
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]  # Get the last time step
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx


tokenized_data = torch.tensor(encode(text), dtype=torch.long)
n = len(tokenized_data)
train_data = tokenized_data[:int(n*0.9)]
val_data = tokenized_data[int(n*0.9):]

train_dataset = CharDataset(train_data, block_size=8)
val_dataset = CharDataset(val_data, block_size=8)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# Create model and move to device
# Initialize model
model = SimpleTransformer(
    vocab_size=vocab_size,
    embed_size=256,
    num_heads=4,
    num_layers=3
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)
print(decode(generated[0].tolist()))