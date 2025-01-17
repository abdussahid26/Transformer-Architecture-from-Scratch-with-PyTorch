# PyTorch Implementation of Transformer

This repository provides an implementation of a **Transformer model** using **PyTorch**, a powerful and flexible deep learning library. The **Transformer architecture**, introduced in the seminal paper *"Attention is All You Need"* by Vaswani et al., has transformed the field of natural language processing (NLP) and has become the foundation of state-of-the-art models like **BERT**, **GPT**, and **T5**. The Transformer leverages **self-attention mechanisms** and **feed-forward networks** to model long-range dependencies in sequential data efficiently.

## Key Features
- **Multi-Head Self-Attention:** Enables the model to attend to multiple parts of the input simultaneously.
- **Positional Encoding:** Encodes sequence order information into the input embeddings.
- **Feed-Forward Network (FFN):** Adds non-linearity and projects attention outputs to higher dimensions.
- **Residual Connections and Layer Normalization:** Ensure stable training and improved gradient flow.
- **Encoder-Decoder Structure:** Implements the standard Transformer framework, separating input encoding from output generation.

![Transformer](https://github.com/abdussahid26/Transformer-Architecture-from-Scratch-with-PyTorch/blob/ee8e2acd47dd7b8fc49d77c55f5012ce77f43c6c/Images/Transformer.png)


## Implementation of Input Embedding

```
lass InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)   

```
