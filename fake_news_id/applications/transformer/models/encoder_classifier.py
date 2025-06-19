import torch
import torch.nn as nn

from applications.transformer.modules.encoder import Encoder


class EncoderClassifier(nn.Module):
    '''
    Custom classifier with a Transformer's Encoder block.
    '''

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_classes: int,
        embed_dim: int,
        n_heads: int,
        ff_hid_dim: int,
        max_length: int,
        pad_idx: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        '''
        Class initializer.

        Args:
            vocab_size: Size of vocabulary.
            n_layers: Number of Encoder layers.
            n_classes: Number of classes for output.
            embed_dim: Size of embedding dimension.
            n_heads: Number of self-attention heads.
            ff_hid_dim: Size of hidden dimension in feed-forward layer of Encoder.
            max_length: Maximum length of vector.
            pad_idx: Index of padding token.
            dropout: p of dropout.
            device: Computing device
        '''
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_hid_dim=ff_hid_dim,
            max_length=max_length,
            dropout=dropout,
            device=device,
        )

        self.linear = nn.Linear(
            in_features=embed_dim,
            out_features=n_classes,
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.to(self.device)

    def forward(self, x: torch.Tensor):
        '''
        Forward pass through architecture.

        Args:
            x: Input tensor.
        '''
        mask = self.mask(x)

        x = self.encoder(x, mask)
        x = self.dropout(x)

        x = x.max(dim=1)[0]

        out = self.linear(x)

        return out
