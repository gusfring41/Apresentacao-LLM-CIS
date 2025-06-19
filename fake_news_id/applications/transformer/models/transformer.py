import torch
import torch.nn as nn

from applications.transformer.modules.encoder import Encoder
from applications.transformer.modules.decoder import Decoder


class Transformer(nn.Module):
    '''
    Transformer architecture with Encoder and Decoder blocks.
    '''

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_pad_idx: int,
        trg_pad_idx: int,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        ff_hid_dim: int,
        max_length: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        '''
        Class initializer.

        Args:
            src_vocab_size: Size of source vocabulary.
            trg_vocab_size: Size of target vocabulary.
            src_pad_idx: Index of source padding token.
            trg_pad_idx: Index of target padding token.
            embed_dim: Size of embedding dimension.
            n_layers: Number of Encoder layers.
            n_heads: Number of self-attention heads.
            ff_hid_dim: Size of hidden dimension in feed-forward layer.
            max_length: Maximum length of vector.
            dropout: p of dropout.
            device: Computing device
        '''
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_hid_dim=ff_hid_dim,
            max_length=max_length,
            dropout=dropout,
            device=device,
        )
        self.decoder = Decoder(
            vocab_size=trg_vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_hid_dim=ff_hid_dim,
            max_length=max_length,
            dropout=dropout,
            device=device,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device

    def src_mask(self, src: torch.Tensor) -> torch.Tensor:
        '''
        Creates mask for source tensor.
        '''
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        '''
        Creates mask for target tensor.
        '''
        N, trg_len = trg.shape

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (
            torch.tril(torch.ones(trg_len, trg_len)).bool().to(self.device)
            & trg_pad_mask
        )

        return trg_pad_mask

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through architecture.

        Args:
            trg: Target tensor.
            src: Source tensor.
        '''
        src_mask = self.src_mask(src)
        trg_mask = self.trg_mask(trg)

        encoded = self.encoder(src, src_mask)
        decoded = self.decoder(trg, encoded, trg_mask, src_mask)

        return decoded
