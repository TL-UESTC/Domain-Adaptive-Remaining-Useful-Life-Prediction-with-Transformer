import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class mymodel(nn.Module):
    def __init__(self, d_model=24, dropout=0.1, nhead=8, nlayers=2, max_len=500) -> None:
        super().__init__()
        self.max_len = max_len
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, 512, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, key_msk, attn_msk=None) -> Tensor:
        """
        return:
            output1: Tensor, extracted features
            output2: Tensor, predicted series
        """
        src = self.pos_encoder(src)
        output1 = self.transformer_encoder(src, attn_msk, key_msk)
        output1 = self.dropout(output1)
        output2 = self.decoder(output1)
        return output1, output2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, feature_num]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)
        

class Discriminator(nn.Module): #D_y
    def __init__(self, in_features=24) -> None:
        super().__init__()
        self.in_features = in_features
        self.li = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor, shape [bts, in_features]
        """
        x = ReverseLayer.apply(x, 1)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.in_features).cuda()
            x = torch.cat((x, pad), 0)
            y = self.li(x)[0].unsqueeze(0)
            return y
        return self.li(x)


class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
        

class backboneDiscriminator(nn.Module): #D_f
    def __init__(self, seq_len, d=24) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.li1 = nn.Linear(d, 1)
        self.li2 = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = ReverseLayer.apply(x, 1)
        out1 = self.li1(x).squeeze(2)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.seq_len).cuda()
            out1 = torch.cat((out1, pad), 0)
            out2 = self.li2(out1)[0].unsqueeze(0)
            return out2
        out2 = self.li2(out1)
        return out2


if __name__ == "__main__":
    pass
    