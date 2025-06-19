import torch
import torch.nn as nn

from ..modules.attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    '''
    Encoder layer with self-attention.
    '''

    def __init__(
        self, embed_dim: int, n_heads: int, ff_hid_dim: int, dropout: float
    ) -> None:
        '''
        Class initializer.

        Args:
            embed_dim: Size of embedding dimension.
            n_heads: Number of self-attention heads.
            ff_hid_dim: Size of hidden dimension in feed-forward layer.
            device: Computing device
        '''
        super().__init__()
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim, n_heads=n_heads, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through architecture.

        Args:
            src: Source tensor.
            mask: Source tensor mask.
        '''
        attention = self.attention(src, src, src, mask)
        x = self.norm_1(attention + self.dropout(src))

        out = self.mlp(x)
        out = self.norm_2(out + self.dropout(x))

        return out


class Encoder(nn.Module):
    '''
    Encoder Block with self-attention.
    '''

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        ff_hid_dim: int,
        max_length: int,
        dropout: float,
        device: str,
    ) -> None:
        '''
        Class initializer.

        Args:
            vocab_size: Size of vocabulary.
            embed_dim: Size of embedding dimension.
            n_layers: Number of Encoder layers.
            n_heads: Number of self-attention heads.
            ff_hid_dim: Size of hidden dimension in feed-forward layer.
            max_length: Maximum length of vector.
            dropout: p of dropout.
            device: Computing device
        '''
        super().__init__()
        self.device = device
        self.scale = embed_dim**0.5

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_length, embed_dim)

        self.blocks = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    ff_hid_dim=ff_hid_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through architecture.

        Args:
            src: Source tensor.
            mask: Source tensor mask.
        '''

        N, seq_len = src.shape

        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        pos_embeddings = self.pos_emb(positions)
        tok_embeddings = self.tok_emb(src) * self.scale
        out = self.dropout(tok_embeddings + pos_embeddings)

        for block in self.blocks:
            out = block(out, mask)

        return out
