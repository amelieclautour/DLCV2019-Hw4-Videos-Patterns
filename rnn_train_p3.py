import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
from datasets import FullLengthVideos
from models import Sequences_p3

def collate_fn(batch):
    inputs = []
    targets = []
    lengths = []

    for i in range(len(batch)):
        inputs.append(batch[i][0])
        targets.append(batch[i][1])
        lengths.append(batch[i][0].size(0))

    inputs = torch.cat(inputs)
    targets = torch.cat(targets)
    lengths = torch.tensor(lengths)

    return inputs, targets, lengths



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 3
    EPOCH = 100

    train_features = torch.load('./rnn_train_feature/train_featuresp3.pt')
    train_labels = torch.load('./rnn_train_feature/train_valsp3.pt')
    train_seq_length =  torch.FloatTensor([len(x) for x in train_features])
    val_features = torch.load('./rnn_train_feature/valid_featuresp3.pt')
    val_labels = torch.load('./rnn_train_feature/valid_valsp3.pt')
    val_seq_length =  torch.FloatTensor([len(x) for x in val_features])


    train_size = len(train_seq_length)
    val_size = len(val_seq_length)
    train_steps = int(np.ceil(train_size / batch_size))
    video_size = 512

    model = Sequences_p3(input_dim=2048, hidden_dim=1024)
    model.load_state_dict(torch.load("./p3.ckpt"))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(EPOCH):

        model.train()
        train_loss = 0.0
        train_acc = 0.0

        index = np.random.permutation(train_size)
        shuf_train_features = [train_features[idx] for idx in index]
        shuf_train_labels = [train_labels[idx] for idx in index]
        shuf_train_lengths = train_seq_length[index]

        for i in range(0, train_size, batch_size):

            inputs, targets = [], []
            for bs in range(batch_size):
                if i + bs >= train_size:
                    break
                frame_num = shuf_train_features[i+bs].size(0)
                frame_idx = sorted(np.random.choice(frame_num, size=video_size, replace=True))

                inputs.append(shuf_train_features[i+bs][frame_idx].to(device))
                targets.append(shuf_train_labels[i+bs][frame_idx])

            # Shape: batch size x video size (3, 512)
            targets = torch.stack(targets).to(device)
            lens = shuf_train_lengths[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs, torch.tensor([video_size for _ in range(len(inputs))]))

            loss = 0.0
            for j in range(len(inputs)):
                loss += criterion(outputs[j], targets[j])
            loss /= len(inputs)
            loss.backward()
            optimizer.step()

            print("Epoch: [{}/{}] | Step: [{}/{}] | Loss={:.5f}".format(epoch+1, EPOCH, (i//batch_size)+1, train_steps, loss.item()), end="")

            predict = torch.max(outputs, 2)[1]
            train_loss += loss.item()
            train_acc += np.sum((predict == targets).cpu().numpy())

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        total = 0

        with torch.no_grad():
            for i in range(val_size):
                inputs = [val_features[i].to(device)]
                targets = val_labels[i].to(device)
                lens = val_seq_length[i:i+1]

                outputs = model(inputs,lens)
                outputs= outputs.squeeze(0)
                loss = criterion(outputs, targets)

                predict = torch.max(outputs, 1)[1]
                val_loss += loss.item()
                val_acc += np.sum((predict == targets).cpu().numpy())
                total += targets.size(0)

        train_loss /= train_steps
        train_acc /= video_size * train_size
        val_loss /= val_size
        val_acc /= total


        torch.save(model.state_dict(), "./checkpoints/p33/{}.ckpt".format(epoch+1))

if __name__ == "__main__":
    os.makedirs("./checkpoints/p33", exist_ok=True)
    main()