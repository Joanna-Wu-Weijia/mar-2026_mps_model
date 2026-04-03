"""
MPS model architecture.

Stage-1  : MultiTask  – three shared transformer encoders trained with a
           co-movement discrimination task (predicts short / mid / long
           correlation class between stock pairs).

Stage-2  : GRU_Predict – bidirectional GRU on top of the frozen encoder
           that outputs a single score for next-day return ranking.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim  = hid_dim
        self.n_heads  = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value):
        B = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy    = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention


class FeedforwardLayer(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        super().__init__()
        self.fc_1    = nn.Linear(hid_dim, pf_dim)
        self.fc_2    = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = FeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, src):
        _src, _ = self.self_attention(src, src, src)
        src     = self.self_attn_layer_norm(src + self.dropout(_src))
        _src    = self.feedforward(src)
        src     = self.ff_layer_norm(src + self.dropout(_src))
        return src


class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderLayers.
    Input shape : (batch, seq_len, hid_dim)
    Output shape: (batch, seq_len, hid_dim)
    """

    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layers  = nn.ModuleList([
            TransformerEncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src)
        return src


# ---------------------------------------------------------------------------
# Stage-1 : Multi-scale encoder with co-movement heads
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Three independent transformer encoders (short / mid / long scale)."""

    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        kwargs = dict(hid_dim=hid_dim, n_layers=n_layers, n_heads=n_heads,
                      pf_dim=pf_dim, dropout=dropout, device=device)
        self.short_encoder  = TransformerEncoder(**kwargs)
        self.middle_encoder = TransformerEncoder(**kwargs)
        self.long_encoder   = TransformerEncoder(**kwargs)

    def forward(self, src):
        """
        src : (batch, seq_len, hid_dim)
        Returns three encodings, each (batch, seq_len, hid_dim).
        """
        return (self.short_encoder(src),
                self.middle_encoder(src),
                self.long_encoder(src))


class MultiTask(nn.Module):
    """
    Stage-1 model.
    Takes a pair of stock feature windows (A, B) and predicts their
    co-movement class at three temporal scales (short / mid / long).

    Input  : srcA, srcB  – (batch, seq_len, n_features)
    Output : three logit tensors of shape (batch, 3)
    """

    def __init__(self, seq_len, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.encoder = Encoder(hid_dim, n_layers, n_heads, pf_dim, dropout, device)

        # Cross-attention: query = one scale's encoding, key/value = all scales
        self.att_short  = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.att_middle = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.att_long   = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        # After concatenating A and B: 2 * seq_len * hid_dim features
        flat_dim = 2 * seq_len * hid_dim

        self.nn_short  = nn.Sequential(nn.Linear(flat_dim, 30), nn.ReLU(), nn.Linear(30, 3))
        self.nn_middle = nn.Sequential(nn.Linear(flat_dim, 30), nn.ReLU(), nn.Linear(30, 3))
        self.nn_long   = nn.Sequential(nn.Linear(flat_dim, 30), nn.ReLU(), nn.Linear(30, 3))

    def forward(self, srcA, srcB):
        sA, mA, lA = self.encoder(srcA)
        sB, mB, lB = self.encoder(srcB)

        # Concatenate along seq-len dim: (batch, 2*seq_len, hid_dim)
        short_enc  = torch.cat([sA, sB], dim=1)
        middle_enc = torch.cat([mA, mB], dim=1)
        long_enc   = torch.cat([lA, lB], dim=1)

        # Global context: all scales
        context = torch.cat([short_enc, middle_enc, long_enc], dim=1)

        short_att,  _ = self.att_short(short_enc,  context, context)
        middle_att, _ = self.att_middle(middle_enc, context, context)
        long_att,   _ = self.att_long(long_enc,    context, context)

        B = srcA.shape[0]
        short_score  = self.nn_short(short_att.view(B, -1))
        middle_score = self.nn_middle(middle_att.view(B, -1))
        long_score   = self.nn_long(long_att.view(B, -1))

        return short_score, middle_score, long_score


# ---------------------------------------------------------------------------
# Stage-2 : GRU predictor on frozen encoder representations
# ---------------------------------------------------------------------------

class GRU_Predict(nn.Module):
    """
    Stage-2 model.
    Takes three encoded representations from the frozen Encoder and produces
    a single scalar score for stock return ranking.

    Input : s_enc, m_enc, l_enc – each (batch, seq_len, hid_dim)
    Output: (batch, 1)
    """

    def __init__(self, seq_len, hid_dim, gru_hidden, gru_layers):
        super().__init__()
        combined_flat = seq_len * 3 * hid_dim   # flatten all three scales
        self.attention = nn.Linear(combined_flat, seq_len * hid_dim)

        self.gru     = nn.GRU(hid_dim, gru_hidden, gru_layers,
                              batch_first=True, bidirectional=True)
        self.linear  = nn.Linear(2 * gru_hidden, 1)
        self.sigmoid = nn.Sigmoid()

        self._hid_dim  = hid_dim
        self._seq_len  = seq_len

    def forward(self, s_enc, m_enc, l_enc):
        B = s_enc.shape[0]

        # (batch, 3*seq_len, hid_dim) → (batch, 3*seq_len*hid_dim)
        enc = torch.cat([s_enc, m_enc, l_enc], dim=1).view(B, -1)

        # Attention bottleneck → (batch, seq_len * hid_dim)
        enc = self.attention(enc)
        enc = enc.view(B, self._seq_len, self._hid_dim)

        output, _ = self.gru(enc)          # (batch, seq_len, 2*gru_hidden)
        out = output[:, -1, :]             # last time-step
        out = self.linear(out)             # (batch, 1)
        out = self.sigmoid(out)
        return out
