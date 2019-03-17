import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RowCNN(nn.Module):
    def __init__(self, num_classes=8, split_size=120):
        super(RowCNN, self).__init__()
        self.window_sizes = [3, 4, 5]
        self.n_filters = 100
        self.num_classes = num_classes
        self.split_size = split_size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.n_filters, [window_size, self.split_size], padding=(window_size - 1, 0))
            for window_size in self.window_sizes
        ])

        self.linear = nn.Linear(self.n_filters * len(self.window_sizes), self.num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        xs = []
        # x = torch.unsqueeze(x, 1) # Might have to do it - [B, CH, R, C]
        # TODO : Very unsure, lot of changes to do
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, R_, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, R_]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2) 
        x = x.view(x.size(0), -1)  
        logits = self.linear(x)
        logits = self.dropout(logits)
        return logits

