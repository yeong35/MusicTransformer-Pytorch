import torch
import torch.nn as nn

class MusicDiscriminator(nn.Module):
    def __init__(self, num_prime = 256):
        super().__init__()
        self.net = nn.Sequential(
            # input shape: (batch_size, num_prime)
            nn.Linear(num_prime, num_prime//2),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//2, num_prime//4),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//4, num_prime//8),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//8, num_prime//16),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//16, num_prime//32),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//32, num_prime//64),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//64, num_prime//128),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//128, 1),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class POP_Classic_Classificator(nn.Module):
    def __init__(self, num_prime = 256):
        super(self).__init__()
        self.net = nn.Sequential(
            # input shape: (batch_size, MAX_LEN)
            nn.Linear(num_prime, num_prime//2),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//2, num_prime//4),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//4, num_prime//8),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//8, num_prime//16),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//16, num_prime//32),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//32, num_prime//64),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Linear(num_prime//64, num_prime//128),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x):
        fx = self.net(x)
        return fx
