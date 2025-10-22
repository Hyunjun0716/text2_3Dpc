import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear
import math


class CrossAttention(Module):
    """
    Cross-attention module for text-to-point conditioning.
    Query: point features, Key/Value: text token embeddings
    """

    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        """
        Args:
            query_dim: Dimension of query (point features)
            context_dim: Dimension of context (text embeddings)
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout probability
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = Linear(query_dim, inner_dim, bias=False)
        self.to_k = Linear(context_dim, inner_dim, bias=False)
        self.to_v = Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None):
        """
        Args:
            x: Query tensor (B, N_points, query_dim)
            context: Context tensor (B, N_tokens, context_dim)
            mask: Optional attention mask (B, N_points, N_tokens)
        Returns:
            Output tensor (B, N_points, query_dim)
        """
        h = self.heads

        # Project to Q, K, V
        q = self.to_q(x)  # (B, N_points, inner_dim)
        k = self.to_k(context)  # (B, N_tokens, inner_dim)
        v = self.to_v(context)  # (B, N_tokens, inner_dim)

        # Reshape for multi-head attention
        # (B, N, inner_dim) -> (B, N, heads, dim_head) -> (B, heads, N, dim_head)
        q = q.reshape(q.size(0), q.size(1), h, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.size(0), k.size(1), h, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.size(0), v.size(1), h, -1).permute(0, 2, 1, 3)

        # Attention: (B, heads, N_points, dim_head) @ (B, heads, dim_head, N_tokens)
        #         -> (B, heads, N_points, N_tokens)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask: (B, N_points, N_tokens) -> (B, 1, N_points, N_tokens)
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        # Apply attention to values: (B, heads, N_points, N_tokens) @ (B, heads, N_tokens, dim_head)
        #                          -> (B, heads, N_points, dim_head)
        out = torch.matmul(attn, v)

        # Reshape back: (B, heads, N_points, dim_head) -> (B, N_points, heads*dim_head)
        out = out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), -1)

        # Project to output
        return self.to_out(out)


class FeedForward(Module):
    """
    Feed-forward network with GELU activation.
    """

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(Module):
    """
    Transformer block with self-attention, cross-attention, and feed-forward.
    """

    def __init__(self, dim, context_dim, heads=8, dim_head=64, dropout=0.0,
                 use_self_attn=True, use_cross_attn=True):
        """
        Args:
            dim: Feature dimension
            context_dim: Context (text) dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout probability
            use_self_attn: Whether to use self-attention
            use_cross_attn: Whether to use cross-attention
        """
        super().__init__()
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn

        if use_self_attn:
            self.self_attn = CrossAttention(dim, dim, heads, dim_head, dropout)
            self.norm1 = nn.LayerNorm(dim)

        if use_cross_attn:
            self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
            self.norm2 = nn.LayerNorm(dim)

        self.ff = FeedForward(dim, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        """
        Args:
            x: Input tensor (B, N, dim)
            context: Context tensor (B, N_ctx, context_dim)
        """
        # Self-attention
        if self.use_self_attn:
            x = x + self.self_attn(self.norm1(x), self.norm1(x))

        # Cross-attention
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)

        # Feed-forward
        x = x + self.ff(self.norm3(x))

        return x


class FiLM(Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies scale and shift based on conditioning.
    """

    def __init__(self, feature_dim, cond_dim):
        """
        Args:
            feature_dim: Dimension of features to modulate
            cond_dim: Dimension of conditioning vector
        """
        super().__init__()
        self.scale_shift = Linear(cond_dim, feature_dim * 2)

    def forward(self, x, cond):
        """
        Args:
            x: Features (B, N, feature_dim) or (B, feature_dim)
            cond: Conditioning (B, cond_dim) or (B, 1, cond_dim)
        Returns:
            Modulated features
        """
        # Ensure cond has batch dimension
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, cond_dim)

        scale_shift = self.scale_shift(cond)  # (B, 1, feature_dim*2)
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each (B, 1, feature_dim)

        return x * (1 + scale) + shift
