from torch import nn
import torch
import numpy as np


class TextLSTM(nn.Module):
    """
    lstm model of text classify.
    """
    def __init__(self, opt):
        self.name = "TextLstm"
        super(TextLSTM, self).__init__()
        self.opt = opt

        self.embedding = nn.Embedding(opt.vocab_size, opt.embed_dim)

        self.lstm = nn.LSTM(input_size=opt.embed_dim,
                            hidden_size=opt.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.linears = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(opt.linear_hidden_size, opt.num_classes),
            # nn.Softmax()
        )

        if opt.embedding_path:
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(opt.embedding_path)))
        #     # self.embedding.weight.requires_grad = False

    def forward(self, x):
        x = self.embedding(x)
        print x
        lstm_out, _ = self.lstm(x)

        # print lstm_out
        # print lstm_out
        out = self.linears(lstm_out[:, -1, :])
        # out = self.linears(x)
        return out


