import torch
import torch.nn as nn
from torch.nn import functional as F

# set device type
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


class CommentaryGPT(nn.Module):
    """
    B -> Batch (batch_size)
    T -> Time-step (block_size)
    C -> Channels (n_embed)
    V -> Vocab-size (vocab_size)
    """

    def __init__(self, config):
        super().__init__()
        global block_size
        global dropout

        block_size = config.block_size
        dropout = config.dropout
        n_embed = config.n_embed
        n_head = config.n_head
        n_layer = config.n_layer
        vocab_size = config.vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.l_norm = nn.LayerNorm(n_embed)
        self.ln_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, targets=None):
        # expected i/p size -- (B, T) [both inputs & targets]
        # expected o/p size -- (B, T, V)
        B, T = inputs.shape

        token_embeddings = self.token_embedding_table(inputs)  # (B, T, C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.l_norm(x)  # (B, T, C)
        logits = self.ln_head(x)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, inputs, max_num_tokens):
        for _ in range(max_num_tokens):
            # crop inputs to the last block_size tokens
            cropped_inp = inputs[:, -block_size:]
            # get predictions
            logits, loss = self(cropped_inp)
            # focus should be only on the last time-step
            logits = logits[:, -1, :]  # (B, C)
            # compute probabilities
            probs = F.softmax(logits, dim=-1)  # (B, V)
            # sample from distribution
            id_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled id to running sequence
            inputs = torch.cat((inputs, id_next), dim=1)

        return inputs


class SelfAttention(nn.Module):
    """
    single self-attention head
    B -> Batch (batch_size)
    T -> Time-step (block_size)
    C -> Channels (n_embed)
    H -> Head (head_size)
    """

    def __init__(self, n_embed, head_size):
        super().__init__()

        self.w_query = nn.Linear(n_embed, head_size, bias=False)
        self.w_key = nn.Linear(n_embed, head_size, bias=False)
        self.w_value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs):
        # expected i/p size -- (B, T, C)
        # expected o/p size -- (B, T, H)
        B, T, C = inputs.shape

        queries = self.w_query(inputs)
        keys = self.w_key(inputs)
        values = self.w_value(inputs)

        # compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)  # (B, T, H) @ (B, H, T) --> (B, T, T)
        attn_scores = attn_scores / (keys.shape[-1] ** 0.5)
        attn_weights = attn_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # perform weighted aggregation of the values
        context_vectors = attn_weights @ values  # (B, T, T) @ (B, T, H) --> (B, T, H)

        return context_vectors


class MultiHeadAttention(nn.Module):
    """
    multiple self-attention heads in parallel
    B -> Batch (batch_size)
    T -> Time-step (block_size)
    C -> Channels (n_embed)
    H -> Head (head_size)
    """

    def __init__(self, n_head, n_embed, head_size):
        super().__init__()

        self.attn_heads = nn.ModuleList([SelfAttention(n_embed, head_size) for _ in range(n_head)])
        self.ln_proj = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # expected i/p size -- (B, T, C)
        # expected o/p size -- (B, T, C)

        x = torch.cat([attn_head(inputs) for attn_head in self.attn_heads], dim=-1)
        x = self.ln_proj(x)
        x = self.dropout(x)

        return x


class FeedForward(nn.Module):
    """
    simple linear layer accompanied by non-linearity
    """

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
    """
    transformer block: communication followed by computation
    """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head

        self.attn_heads = MultiHeadAttention(n_head, n_embed, head_size)
        self.feed_fwd = FeedForward(n_embed)
        self.l_norm_1 = nn.LayerNorm(n_embed)
        self.l_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, inputs):
        x = inputs + self.attn_heads(self.l_norm_1(inputs))
        x = x + self.feed_fwd(self.l_norm_2(x))

        return x
