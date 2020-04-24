import torch.nn as nn
import torchvision.models as models
import torch

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

class LongShortTermMemory_RNN(nn.Module):
    def __init__(self, input_size, hidden_size=512, n_layers=1, dropout=0):
        super(LongShortTermMemory_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=n_layers,dropout=dropout, bidirectional=False).cuda()

        self.lin1 = nn.Linear(self.hidden_size,1024)
        self.lin2 = nn.Linear(1024,256)
        self.lin3 = nn.Linear(256, 11)
        self.softmax = nn.Softmax(1)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.bn_2 = nn.BatchNorm1d(256)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, padded_sequence, input_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths,batch_first=True)
        outputs, (hn,cn) = self.lstm(packed)
        hidden_output = hn[-1]
        outputs = self.relu(self.bn_1(self.lin1(hidden_output)))
        outputs = self.relu(self.bn_2(self.lin2(outputs)))
        results = self.softmax(self.lin3(outputs))
        return results, hidden_output, outputs

class LSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, n_layers=1, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
            dropout=dropout, bidirectional=True, batch_first=True)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc_2 = nn.Linear(int(self.hidden_size), 11)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, padded_sequence, input_length, hidden=None):
        packed = nn.utils.rnn.pack_padded_sequence(padded_sequence, input_length, batch_first=True)
        outputs, (hn, cn) = self.lstm(packed, hidden)
        # outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # print("outputs",outputs.size())
        # print("cn",cn.size())
        hidden_output = hn[-1]
        # hidden_output = torch.mean(outputs,1)
        outputs = self.fc_1(hidden_output)
        outputs = self.bn_0(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)


        outputs = self.fc_2(outputs)
        return outputs

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        model = models.vgg16(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        """
            input shape: 224 x 224
            output shape: batch size x 512 x 7 x 7
        """
        output = self.feature(x)
        return output

class crossLoss(nn.Module):
    def __init__(self):
        super(crossLoss, self).__init__()

    def forward(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = 0
        batch_size = len(predict)

        for i in range(batch_size):
            print(predict[i], target[i])
            partial_loss = loss_fn(predict[i], target[i])
            loss += partial_loss
        loss = loss / batch_size
        return loss