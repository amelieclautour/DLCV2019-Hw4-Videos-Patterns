import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torchvision.models as models



class RNNVideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNVideoClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_dim, 11)
        )


    def forward(self, x, length):
        # Sort the lengths in descending order
        idx = torch.argsort(-length)
        # Keep the original index
        desort_idx = torch.argsort(idx)
        # Sort x and length in descending order
        x = [x[i] for i in idx]
        length = length[idx]

        x_padded = rnn.pad_sequence(x, batch_first=True)
        x_packed = rnn.pack_padded_sequence(x_padded, length, batch_first=True)
        gru_outputs, _ = self.gru(x_packed)
        gru_outputs, _ = rnn.pad_packed_sequence(gru_outputs, batch_first=True)

        gru_outputs = gru_outputs[torch.arange(gru_outputs.size(0)), length-1]

        outputs =  self.fc(gru_outputs)
        outputs = outputs[desort_idx]

        return outputs

class Sequences_p3(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Sequences_p3, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_dim, 11)
        )


    def forward(self, x, length):
        # Sort the lengths in descending order
        idx = torch.argsort(-length)
        # Keep the original index
        desort_idx = torch.argsort(idx)
        # Sort x and length in descending order
        x = [x[i] for i in idx]
        length = length[idx]

        x_padded = rnn.pad_sequence(x, batch_first=True)
        x_packed = rnn.pack_padded_sequence(x_padded, length, batch_first=True)
        gru_outputs, _ = self.gru(x_packed)
        gru_outputs, _ = rnn.pad_packed_sequence(gru_outputs, batch_first=True)

        outputs =  self.fc(gru_outputs)
        outputs = outputs[desort_idx]

        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_dim, 11)
        )


    def forward(self, x, length):
        # Sort the lengths in descending order
        idx = torch.argsort(-length)
        # Keep the original index
        desort_idx = torch.argsort(idx)
        # Sort x and length in descending order
        x = [x[i] for i in idx]
        length = length[idx]

        x_padded = rnn.pad_sequence(x, batch_first=True)
        x_packed = rnn.pack_padded_sequence(x_padded, length, batch_first=True)
        gru_outputs, _ = self.gru(x_packed)
        gru_outputs, _ = rnn.pad_packed_sequence(gru_outputs, batch_first=True)

        outputs =  self.fc(gru_outputs)
        outputs = outputs[desort_idx]

        return outputs

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        output = self.feature(x)
        output = output.view(-1,2048)
        return output

class NeuralNetwork(nn.Module):
    def __init__(self, feature_size):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(feature_size,4096)
        self.lin2 = nn.Linear(4096,1024)
        self.lin3 = nn.Linear(1024, 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.batch1 = nn.BatchNorm1d(4096)
        self.batch2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.relu(self.batch1(self.lin1(x))) # same as relu output
        x = self.relu(self.batch2(self.lin2(x)))
        y_pred = self.softmax(self.lin3(x))
        return y_pred,x