import torch
from torch import nn

    #总声压级模型
class SPLModel(nn.Module):
    def __init__(self, in_nc=5, nc=400, num_bins=1):
        super(SPLModel, self).__init__()
        self.num_bins = num_bins
        self.hidden1 = nn.Linear(in_nc, nc)
        self.dropout1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(nc, nc)
        self.dropout2 = nn.Dropout(0.2)
        self.hidden3 = nn.Linear(nc, nc)
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, self.num_bins)
    def forward(self, inp):
        x = self.relu(self.hidden1(inp))
        x = self.dropout1(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout3(x)
        out = self.out(x)
        output = out.view(-1, self.num_bins)
        return output
    #1/3声压级模型
class Octave_1_3_Model(nn.Module):
    def __init__(self, in_nc=5, nc=400,  num_bins=28):
        super(Octave_1_3_Model, self).__init__()
        self.num_bins = num_bins
        self.hidden1 = nn.Linear(in_nc, nc)
        self.dropout1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(nc, nc)
        self.dropout2 = nn.Dropout(0.2)
        self.hidden3 = nn.Linear(nc, nc)
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, self.num_bins)
    def forward(self, inp):
        x = self.relu(self.hidden1(inp))
        x = self.dropout1(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout3(x)
        out = self.out(x)
        output = out.view(-1, self.num_bins)
        return output
    #线谱模型
class LineOctaveModel(nn.Module):
    def __init__(self, in_nc=5, nc=400,  num_bins=2501):
        super(LineOctaveModel, self).__init__()
        self.num_bins = num_bins
        self.hidden1 = nn.Linear(in_nc, nc)
        self.dropout1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(nc, nc)
        self.dropout2 = nn.Dropout(0.2)
        self.hidden3 = nn.Linear(nc, nc)
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, self.num_bins)
    def forward(self, inp):
        x = self.relu(self.hidden1(inp))
        x = self.dropout1(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout3(x)
        out = self.out(x)
        output = out.view(-1, self.num_bins)
        return output
    #声品质模型
class SoundqualityModel(nn.Module):
    def __init__(self, in_nc=7, nc=400, mode_nc=4, num_bins=1):
        super(SoundqualityModel, self).__init__()
        self.num_bins = num_bins
        self.mode_nc = mode_nc
        self.hidden1 = nn.Linear(in_nc, nc)
        self.dropout1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(nc, nc)
        self.dropout2 = nn.Dropout(0.2)
        self.hidden3 = nn.Linear(nc, nc)
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, mode_nc * num_bins)
    def forward(self, inp):
        x = self.relu(self.hidden1(inp))
        x = self.dropout1(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout3(x)
        out = self.out(x)
        output = out.view(-1, self.mode_nc, self.num_bins)
        return output
        