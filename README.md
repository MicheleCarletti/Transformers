# Transformers

A simple example demonstrating the implementation of Transformer models
using PyTorch.

## Overview

This repository contains a minimal example of building and training a
simple Transformer model for next-word prediction tasks. It also
includes visualization of attention weights.

## Files

-   `transformer_example.py`: Main Python script implementing the
    Transformer model.
-   `data/data.txt`: Training data file (plain text).
-   `data/test.txt`: Test input file for predictions and attention
    visualization.

## Code Breakdown

### Main Components:

-   **Tokenization Utilities**: `encode(s)`, `decode(l)` for converting
    between words and indices.
-   **Positional Encoding**: Implements sinusoidal positional encodings.
-   **Transformer Block**: Multi-head self-attention and feed-forward
    layers.
-   **Simple Transformer Model**: Token embedding, positional encoding,
    transformer block, and output projection.
-   **Training Loop**: Simple next-word prediction training on tokenized
    text data.
-   **Visualization**: Attention weights visualization for
    interpretability.

### Key Functions

-   `predict_next_tokens(model, file_path, n_pred=2)`: Generates the
    next `n_pred` tokens given an input file.
-   `visualize_attention(model, file_path)`: Visualizes attention weight
    matrices of the model for given input tokens.

## Usage

1.  **Prepare data**:

    -   Place your training corpus in `data/data.txt`.
    -   Place your test input sequence in `data/test.txt`.

2.  **Run the script**: `bash     python transformer_example.py`

3.  **Output**:

    -   The script will print predictions for the next tokens.
    -   Attention weight matrices will be visualized using matplotlib.

## Requirements

-   Python 3.7+
-   PyTorch
-   NumPy
-   Matplotlib

Install dependencies via:

``` bash
pip install torch numpy matplotlib
```

## Example Output

    Epoch 0: Loss = 4.3210
    Epoch 20: Loss = 2.5674
    ...
    Input text: 'the quick brown'
    Generated text: the quick brown fox jumps

## Visualization

Attention heatmaps will display how each token attends to others in the
sequence, for each attention head.

## License

MIT License
