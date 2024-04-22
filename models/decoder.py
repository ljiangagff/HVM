import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch import nn
from torch import einsum, nn

from .positionEncoding import RotaryEmbedding, SwiGLU, Residual, apply_rotary_pos_emb, PositionalEncoding
from einops import rearrange, repeat


class FTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.

    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError(
                    "cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            return output

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat(
                [cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class FTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            see FTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """

        if self.training:
            return super().forward(
                tgt,
                memory,
                tgt_mask=generate_square_subsequent_mask(
                    tgt.size(0), tgt.device),
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        tgt_last_tok = tgt[-1:, :, :]

        # self attention part
        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        if memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


def generate_square_subsequent_mask(sz, device) -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask


class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, device, output_dim, embed_dim=256, hidden_dim=256, enc_hid_dim=256, dec_hid_dim=256, num_layers=1, dropout=0.1):
        super(LSTMDecoderWithAttention, self).__init__()
        self.device = device
        # self.embedding = nn.Embedding(output_dim, embed_dim)
        self.attention = Attention(enc_hid_dim, dec_hid_dim).to(device)
        self.lstm = nn.LSTM(embed_dim+enc_hid_dim,
                            hidden_dim, num_layers, dropout=dropout).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, tgt, memory, cache=None):
        trg_len = tgt.shape[0]
        batch_size = tgt.shape[1]

        outputs = torch.zeros(trg_len, batch_size,
                              self.lstm.hidden_size).to(self.device)
        hidden, cell = torch.zeros(1, batch_size, self.lstm.hidden_size).to(
            self.device), torch.zeros(1, batch_size, self.lstm.hidden_size).to(self.device)

        for t in range(1, trg_len):
            output, hidden, cell = self.decode(
                tgt[t, :, :].unsqueeze(0), hidden, cell, memory)
            outputs[t] = output
            # teacher_force = np.random.random() < teacher_forcing_ratio
            # top1 = output.argmax(1)

        if self.training:
            return outputs
        else:
            return outputs, cache

    def decode(self, embedded, hidden, cell, encoder_outputs):
        # input = input.unsqueeze(0)
        # embedded = self.dropout(self.embedding(input))
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(
            1), encoder_outputs.transpose(0, 1))
        # attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_applied = attn_applied.transpose(0, 1)
        lstm_input = torch.cat((embedded, attn_applied), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # prediction = self.fc_out(output.squeeze(0))
        return output.squeeze(0), hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(0), 1)
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs.transpose(0, 1)), dim=2)))
        attention = F.softmax(torch.sum(energy, dim=2), dim=1)
        return attention


class TextEncoder(nn.Module):
    def __init__(self, embeddingLayer, device, dim=512, heads=8, num_layers=6, pad_id=0) -> None:
        super().__init__()
        self.layers = []
        self.embedding = embeddingLayer
        self.text_cls_token = nn.Parameter(torch.randn(dim))
        self.cls_norm = nn.LayerNorm(dim)
        self.pad_id = pad_id
        for _ in range(num_layers):
            self.layers.append(
                Residual(TextEncoderLayer(dim=dim, dim_head=int(
                    dim/heads), heads=heads).to(device))
            )

    def forward(self, text):
        batch = text.size(0)
        seq = text.size(1)

        text_tokens = self.embedding(text)
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)
        # create specific mask for text cls token at the end
        # to prevent it from attending to padding
        cls_mask = rearrange(text != self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)
        # go through unimodal layers
        for attn_ff in self.layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)
        # get text cls token
        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        text_embeds = self.cls_norm(text_cls_tokens)
        return text_tokens, text_embeds


class TextEncoderLayer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head,
                           dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(
            dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings
        self.mask = None
        self.pos_emb = None

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n].to(device)

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.mask = mask
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)

        pos_emb = self.rotary_emb(n, device=device)
        self.pos_emb = pos_emb
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device, h = x.shape[1], x.device, self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        # split heads
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        # rotary embeddings
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
        q = q * self.scale
        sim = einsum("b h i d, b j d -> b h i j", q, k)
        # causal mask
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        # aggregate values and merge heads
        out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


class NoneTextEncoder(nn.Module):
    def __init__(self, device, embeddingLayer, hdim) -> None:
        super().__init__()
        self.embedding = embeddingLayer
        self.pos = PositionalEncoding(hdim).to(device)

    def forward(self, text):
        text_tokens = self.embedding(text)
        return self.pos(text_tokens), None


