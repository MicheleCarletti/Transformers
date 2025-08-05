"""
@author Michele Carletti
A simple example on Transformer models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



# Encode function
def encode(s):
    return [stoi[word] for word in s.strip().split()]

# Decode function
def decode(l):
    return ' '.join([itos[int(i)] for i in l])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, d_model * 6),
            nn.ReLU(),
            nn.Linear(d_model * 6, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.ln1(x + attn_output)
        x = self.ln2(x + self.fcn(x))
        return x, attn_weights
    
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nheads=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer_block = TransformerBlock(d_model, nheads)
        self.fcn_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x, attn_weights = self.transformer_block(x)
        logits = self.fcn_out(x)
        return logits, attn_weights

def visualize_attention(model, file_path):
    model.eval()
    with open(file_path, 'r') as f:
        input_text = f.read().strip()
    
    input_tokens = encode(input_text)
    x_input = torch.tensor([input_tokens], dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(x_input)

    # Get the predicted token
    predicted_idx = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1)
    # Append predicted token to X_input sequence
    X_input = torch.cat([x_input, predicted_idx], dim=1)  # Shape: (1, seq_len + 1)

    # Re-run forward pass to get attention weights for updated sequence
    with torch.no_grad():
        _, attn_weights = model(x_input)

    # attn_weights: shape (batch_size, num_heads, seq_len, seq_len)
    attn = attn_weights.detach().numpy()

    batch_idx = 0
    num_heads = attn.shape[1]
    seq_len = attn.shape[2]

    # Decode tokens for axis labels
    tokens = [itos[i.item()] for i in x_input[batch_idx]]

    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 6))

    if num_heads == 1:
        axes = [axes]  # Ensure iterable

    for head in range(num_heads):
        ax = axes[head]
        im = ax.imshow(attn[batch_idx, head], cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_title(f'Head {head}')

        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))

        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)

        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
    
    # Show colorbar as intensity legend
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Attention weight intensity")
    plt.show()  

def predict_next_tokens(model, file_path, n_pred=2):
    model.eval()
    with open(file_path, 'r') as f:
        input_text = f.read().strip()

    print(f"Input text: '{input_text}'")

    # Encode initial input
    input_tokens = encode(input_text)
    x_input = torch.tensor([input_tokens], dtype=torch.long)

    generated_tokens = input_tokens.copy()

    for _ in range(n_pred):
        with torch.no_grad():
            logits, _ = model(x_input)

        # Get prediction for the last token in sequence
        next_token_logits = logits[0, -1]
        predicted_idx = torch.argmax(next_token_logits).item()

        # Append prediction
        generated_tokens.append(predicted_idx)

        # Update X_input
        x_input = torch.tensor([generated_tokens], dtype=torch.long)

    generated_text = decode(generated_tokens)
    print(f"Generated text: {generated_text}")


if __name__ == "__main__":

    test_file = 'data/test.txt'
    # Sample data 
    with open('data/data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Create the volabulary
    words = text.strip().split()
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for i, word in enumerate(vocab)}

    # Dataset preparation (next char prediction)
    data = torch.tensor(encode(text), dtype=torch.long)
    X = data[:-1].unsqueeze(0)  # Input sequence (batch_size, sequence_length)
    y = data[1:].unsqueeze(0)   # Target sequence (batch_size, sequence_length)

    model = SimpleTransformer(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        optimizer.zero_grad()
        logits, attn_weights = model(X)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        

    # Test and Visualize Attention weights
    predict_next_tokens(model, test_file, 4)
    visualize_attention(model, test_file)
    

    
    



