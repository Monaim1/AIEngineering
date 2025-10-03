# GPT-Karpathy

A PyTorch implementation of a Transformer model with Multi-Head Attention (MHA), inspired by Andrej Karpathy's work on language models.
groing from a simple Bigram model to implementing MHA transformer

## Project Structure

- `MHATransformer.py`: Main implementation of the Transformer model with Multi-Head Attention
- `Colab - MHATransformer.ipynb`: Colab version of the MHATransformer.py script. contains results after training

## Features

- Multi-Head Self-Attention mechanism
- Transformer architecture implementation
- Interactive Colab notebook for easy experimentation


## Getting Started

1. Clone this repository:

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt  # If you have a requirements file
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "Colab - MHATransformer.ipynb"
   ```
   
   Or open it directly in Google Colab.


## Model Architecture

The model implements a standard Transformer architecture with:
- Multi-Head Self-Attention
- Position-wise Feed-Forward Networks
- Layer Normalization
- Residual Connections
