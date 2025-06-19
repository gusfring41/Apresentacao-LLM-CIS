import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    '''
    Self-attention layer for Transformer.
    '''

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        '''
        Class initializer.

        Args:
            vocab_size: Size of vocabulary.
            n_layers: Number of Encoder layers.
            n_classes: Number of classes for output.
            ff_hid_dim: Size of hidden dimension in feed-forward layer.
            embed_dim: Size of embedding dimension.
            n_heads: Number of self-attention heads.
            max_length: Maximum length of vector.
            pad_idx: Index of padding token.
            dropout: p of dropout.
            device: Computing device
        '''
        super().__init__()
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = embed_dim**0.5

        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        '''
        Forward pass through architecture.

        Args:
            q: Tensor of queries.
            k: Tensor of keys.
            v: Tensor of values.
            mask: Energy mask.
        '''
        N = q.shape[0]

        Q = self.queries(q)
        K = self.keys(k)
        V = self.values(v)

        Q = Q.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = (Q.matmul(K.permute(0, 1, 3, 2))) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20)

        attention = energy.softmax(-1)

        x = self.dropout(attention).matmul(V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(N, -1, self.embed_dim)
        x = self.proj(x)

        return x
