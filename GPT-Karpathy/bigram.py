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




class BigramlanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # used to pluck out the embedding of each input idx
        self = self.to(device)  # Move model to device

    def forward(self, idx, targets=None):
        # Ensure input is on the same device as the model
        idx = idx.to(device)
        if targets is not None:
            targets = targets.to(device)

        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C) (batch, time, channel) (4, 8, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets) # cross entripy exepects (B,C, T)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        idx = idx.to(device)  # Ensure input is on the same device as the model
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # Focus on the last time step
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
blm = BigramlanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(blm.parameters(), lr=1e-3)

# Training
num_epochs = 5

for epoch in range(num_epochs):
    # Training phase
    blm.train()  # Set the model to training mode
    train_loss = 0.0
    train_batches = 0
    
    for x_batch, y_batch in train_loader:
        # Move data to the same device as model
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        _, loss = blm(x_batch, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
    
    # Calculate average training loss for the epoch
    avg_train_loss = train_loss / train_batches
    
    # Validation phase
    blm.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():  # No need to track gradients during validation
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, loss = blm(x_batch, y_batch)
            val_loss += loss.item()
            val_batches += 1
    
    # Calculate average validation loss
    avg_val_loss = val_loss / val_batches
    
    print(f'Epoch {epoch + 1}/{num_epochs}, ' 
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}')

# Generate some text after training
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nGenerated text:")
print('-' * 50)
print(decode(blm.generate(context, max_new_tokens=500)[0].cpu().numpy()))
print('-' * 50)