import torch
import torch.nn as nn

from ..modules.attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    '''
    Decoder layer with self-attention.
    '''

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_hid_dim: int,
        dropout: float,
    ) -> None:
        '''
        Class initializer.

        Args:
            embed_dim: Size of embedding dimension.
            n_heads: Number of self-attention heads.
            ff_hid_dim: Size of hidden dimension in feed-forward layer.
            dropout: p of dropout.
        '''

        super().__init__()
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim, n_heads=n_heads, dropout=dropout
        )
        self.joint_attention = MultiHeadAttention(
            embed_dim=embed_dim, n_heads=n_heads, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg: torch.Tensor,
        src: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Forward pass through architecture.

        Args:
            trg: Target tensor.
            src: Source tensor.
            trg_mask: Target tensor mask.
            src_mask: Source tensor mask.
        '''
        trg_attention = self.attention(trg, trg, trg, trg_mask)
        trg = self.norm_1(trg + self.dropout(trg_attention))

        joint_attention = self.attention(trg, src, src, src_mask)
        trg = self.norm_2(trg + self.dropout(joint_attention))

        out = self.mlp(trg)
        out = self.norm_2(trg + self.dropout(out))

        return out


class Decoder(nn.Module):
    '''
    Decoder block with self-attention
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
                DecoderLayer(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    ff_hid_dim=ff_hid_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg: torch.Tensor,
        src: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Forward pass through architecture.

        Args:
            trg: Target tensor.
            src: Source tensor.
            trg_mask: Target tensor mask.
            src_mask: Source tensor mask.
        '''
        N, trg_len = trg.shape

        positions = torch.arange(0, trg_len).expand(N, trg_len).to(self.device)
        pos_embeddings = self.pos_emb(positions)
        tok_embeddings = self.tok_emb(trg) * self.scale
        trg = self.dropout(tok_embeddings + pos_embeddings)

        for block in self.blocks:
            trg = block(trg, src, trg_mask, src_mask)

        out = self.fc(trg)
        return out
