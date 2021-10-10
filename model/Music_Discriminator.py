import torch
import torch.nn as nn

class MusicDiscriminator(nn.Module):
    def __init__(self, primer_num = 256):
        super(self).__init__()
        self.net = nn.Sequential(
            # input shape: (batch_size, MAX_LEN)
            nn.Linear(primer_num, primer_num//2),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//2, primer_num//4),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//4, primer_num//8),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//8, primer_num//16),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//16, primer_num//32),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//32, primer_num//64),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//64, primer_num//128),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//128, primer_num//256),
            nn.LeakyReLU(0.3, inplace=True),

        )

    def forward(self, x):
        fx = self.net(x)
        return fx

class POP_Classic_Classificator(nn.Module):
    def __init__(self, primer_num = 256):
        super(self).__init__()
        self.net = nn.Sequential(
            # input shape: (batch_size, MAX_LEN)
            nn.Linear(primer_num, primer_num//2),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//2, primer_num//4),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//4, primer_num//8),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//8, primer_num//16),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//16, primer_num//32),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//32, primer_num//64),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(primer_num//64, primer_num//128),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x):
        fx = self.net(x)
        return fx
