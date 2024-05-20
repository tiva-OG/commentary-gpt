import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
dropout = 0.2


class SelfAttention(nn.Module):
    """
    single self-attention head
    """

    def __init__(self, n_embed, dim_qk, dim_v):
        super().__init__()
        self.dim_qk = dim_qk

        self.W_query = nn.Linear(n_embed, dim_qk, bias=False)
        self.W_key = nn.Linear(n_embed, dim_qk, bias=False)
        self.W_value = nn.Linear(n_embed, dim_v, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs):
        # B -> batch_size
        # T -> block_size
        # C -> n_embed
        # k -> dim_qk
        # v -> dim_v

        B, T, C = inputs.shape
        queries = self.W_query(inputs)  # (B, T, k)
        keys = self.W_key(inputs)  # (B, T, k)
        values = self.W_value(inputs)  # (B, T, v)
        # compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)  # (B, T, k) @ (B, k, T) --> (B, T, T)
        attn_scores = attn_scores / (self.dim_qk ** 2)
        attn_weights = attn_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # perform weighted aggregation of the values
        context_vectors = attn_weights @ values  # (B, T, T) @ (B, T, dim_v) --> (B, T, dim_v)

        return context_vectors


class MultiHeadAttention(nn.Module):
    """
    multiple self-attention heads in parallel
    """

    def __init__(self, n_head, n_embed, dim_qk, dim_v):
        super().__init__()
        self.dim_qk = dim_qk

        self.attn_heads = nn.ModuleList([SelfAttention(n_embed, dim_qk, dim_v) for _ in range(n_head)])

        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        context_vectors = torch.cat([attn_head(inputs) for attn_head in self.attn_heads], dim=-1)
        x = self.projection(context_vectors)
        x = self.dropout(x)

        return x


class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        return self.net(inputs)


class Block(nn.Module):

    def __init__(self, n_head, n_embed, dim_qk, dim_v):
        super().__init__()
        self.attn_heads = MultiHeadAttention(n_head, n_embed, dim_qk, dim_v)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, inputs):
        x = inputs + self.attn_heads(self.layer_norm_1(inputs))
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x
