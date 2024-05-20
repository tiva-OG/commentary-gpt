"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from src.model.components import Block


@dataclass
class GPTConfig:
    # hyperparameters
    batch_size: int = 4
    block_size: int = 8
    vocab_size: int = 102
    n_embed: int = 16
    n_head: int = 2
    n_blocks: int = 4
    dim_qk: int = 8
    dim_v: int = 8

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CommentaryGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.blocks = nn.Sequential(
            *[Block(config.n_head, config.n_embed, config.dim_qk, config.dim_v) for _ in range(config.n_blocks)])
        self.layer_norm = nn.LayerNorm(config.n_embed)
        self.linear_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, inputs, targets=None):
        # B -> batch_size
        # T -> block_size
        # C -> n_embed
        # V -> vocab_size

        B, T = inputs.shape
        token_embeddings = self.token_embedding_table(inputs)  # (B, T, C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.config.device))  # (T, C)

        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear_head(x)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_num_tokens):
        for _ in range(max_num_tokens):
            # crop inputs to the last block_size tokens
            cropped_inputs = inputs[:, -self.config.block_size:]
            # get predictions
            logits, loss = self(cropped_inputs)
            # focus should be only on the last time-step
            logits = logits[:, -1, :]  # (B, C)
            # compute probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            id_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled id to running sequence
            inputs = torch.cat((inputs, id_next), dim=1)

        return inputs
