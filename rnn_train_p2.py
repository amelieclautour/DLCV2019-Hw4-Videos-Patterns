import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
from models import RNNVideoClassifier



train_features = np.array(torch.load('./rnn_train_feature/train_rnn_features.pt'))
train_labels = torch.load('./rnn_train_feature/train_rnn_vals.pt').type(torch.LongTensor)
train_seq_length =  torch.LongTensor([len(x) for x in train_features])


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 64
    EPOCH = 100


    train_size = len(train_features)
    train_steps = int(np.ceil(train_size / batch_size))

    model = RNNVideoClassifier(input_dim=2048, hidden_dim=1024)
    model.load_state_dict(torch.load("./p2.ckpt"))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(EPOCH):

        model.train()
        train_loss = 0.0
        train_acc = 0.0

        index = np.random.permutation(train_size)
        shuf_train_features = [train_features[idx] for idx in index]
        shuf_train_labels = train_labels[index]
        shuf_train_lengths = train_seq_length[index]

        for i in range(0, train_size, batch_size):
            inputs = [data.to(device) for data in shuf_train_features[i:i+batch_size]]
            targets = shuf_train_labels[i:i+batch_size].to(device)
            lens = shuf_train_lengths[i:i+batch_size]

            optimizer.zero_grad()

            outputs = model(inputs, lens)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            print("\rEpoch: [{}/{}] | Step: [{}/{}] | Loss={:.5f}".format(epoch+1, EPOCH, (i//batch_size)+1, train_steps, loss.item()), end="")

            predict = torch.max(outputs, 1)[1]
            train_loss += loss.item()
            train_acc += np.sum((predict == targets).cpu().numpy())

        model.eval()
        val_loss = 0.0
        val_acc = 0.0



        torch.save(model.state_dict(), "./checkpoints/p2der/{}.ckpt".format(epoch+1))

if __name__ == "__main__":
    os.makedirs("./checkpoints/p2der", exist_ok=True)
    main()