import torch
import torch.nn as nn


# ====================== BiLSTMEncoder ======================
class BiLSTMEncoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, enc_hidden_dim, pad_idx, num_layers=1, dropout=0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=enc_hidden_dim,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        outputs, (h, c) = self.lstm(embedded)

        return outputs, (h, c)


# ====================== HiddenBridge (Bi->Uni LSTM) ======================
class HiddenBridge(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, num_layers=1):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.num_layers = num_layers

        self.W_h = nn.ModuleList(
            [
                nn.Linear(2 * enc_hidden_dim, dec_hidden_dim, bias=True)
                for _ in range(num_layers)
            ]
        )
        self.W_c = nn.ModuleList(
            [
                nn.Linear(2 * enc_hidden_dim, dec_hidden_dim, bias=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, h, c):
        num_layers_total = h.size(0)
        bs = h.size(1)
        assert (
            num_layers_total == 2 * self.num_layers
        ), f"Expected {2*self.num_layers}, got {num_layers_total}"

        h = h.view(self.num_layers, 2, bs, self.enc_hidden_dim)
        c = c.view(self.num_layers, 2, bs, self.enc_hidden_dim)

        new_h, new_c = [], []
        for i in range(self.num_layers):
            h_fw = h[i, 0, :, :]
            h_bw = h[i, 1, :, :]
            h_cat = torch.cat([h_fw, h_bw], dim=-1)

            c_fw = c[i, 0, :, :]
            c_bw = c[i, 1, :, :]
            c_cat = torch.cat([c_fw, c_bw], dim=-1)

            proj_h = torch.tanh(self.W_h[i](h_cat))
            proj_c = torch.tanh(self.W_c[i](c_cat))

            new_h.append(proj_h.unsqueeze(0))
            new_c.append(proj_c.unsqueeze(0))

        new_h = torch.cat(new_h, dim=0)
        new_c = torch.cat(new_c, dim=0)
        return new_h, new_c


# ====================== Decoder (no attention) ======================
class LSTMDecoderNoAttention(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, dec_hidden_dim, pad_idx, num_layers=1, dropout=0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=dec_hidden_dim,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )

        self.layernorm = nn.LayerNorm(dec_hidden_dim)
        self.fc_out = nn.Linear(dec_hidden_dim, vocab_size)

        self.dec_hidden_dim = dec_hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

    def forward(self, tgt_ids, hidden, cell):
        embedded = self.embedding(tgt_ids)

        dec_out, (h, c) = self.lstm(embedded, (hidden, cell))

        ln_out = self.layernorm(dec_out)
        logits = self.fc_out(ln_out)

        return logits, (h, c)

    def forward_step(self, cur_tokens, hidden, cell):
        emb = self.embedding(cur_tokens).unsqueeze(1)
        dec_out, (h, c) = self.lstm(emb, (hidden, cell))
        ln_out = self.layernorm(dec_out)
        logits = self.fc_out(ln_out).squeeze(1)

        return logits, (h, c)


# ====================== Итоговая Seq2Seq ======================
class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        enc_hidden_dim,
        dec_hidden_dim,
        pad_idx,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            enc_hidden_dim=enc_hidden_dim,
            pad_idx=pad_idx,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.bridge = HiddenBridge(
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            num_layers=num_layers,
        )

        self.decoder = LSTMDecoderNoAttention(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dec_hidden_dim=dec_hidden_dim,
            pad_idx=pad_idx,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.pad_idx = pad_idx

    def forward(self, src_ids, tgt_ids):
        encoder_outputs, (h_enc, c_enc) = self.encoder(src_ids)
        dec_h, dec_c = self.bridge(h_enc, c_enc)

        logits, _ = self.decoder(tgt_ids, dec_h, dec_c)
        return logits

    def forward_encoder(self, src_ids):
        return self.encoder(src_ids)

    def init_decoder_state(self, h_enc, c_enc):
        return self.bridge(h_enc, c_enc)
