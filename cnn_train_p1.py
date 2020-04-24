import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn

from models import Resnet50
from models import NeuralNetwork
from videos_read import Short_Vid
from videos_read import get_Vid_List


# loading features extracted by pretrain model
train_features = torch.load('./cnn_train_feature/train_cnn_features.pt').view(-1,2048)
valid_features = torch.load('./cnn_train_feature/valid_cnn_features.pt').view(-1,2048)
train_val = torch.load('./cnn_train_feature/train_cnn_val.pt').type(torch.LongTensor)
valid_val = torch.load('./cnn_train_feature/valid_cnn_val.pt').type(torch.LongTensor)
print("train_features",train_features.shape)
print("train_val",train_val.shape)
print("valid_features",valid_features.shape)
print("valid_val",valid_val.shape)

# model, optimzer, loss function
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
feature_size = 2048
model = NeuralNetwork(feature_size).to(device)
model = torch.load("./best_cnn.pth")
optimizer = torch.optim.Adam(model.parameters(), lr=0.000025)
loss_function = nn.CrossEntropyLoss()


# some training parameters
batch_size = 64
num_epoch = 100
total_length = len(train_features)
max_accuracy = 0
logfile = open('log.txt', 'w')
now = datetime.datetime.now()
logfile.writelines("start training at:"+str(now)+"\n")
logfile.flush()


# start training
model.train()
train_loss = []
valid_acc = []
for epoch in range(num_epoch):
    logfile.writelines("Epoch:"+str(epoch+1)+"\n")
    logfile.flush()
    print("Epoch:", epoch+1)
    total_loss = 0.0
    num_batch = 0

    # shuffle data
    perm_index = torch.randperm(total_length)
    train_features_sfl = train_features[perm_index]
    train_val_sfl = train_val[perm_index]

    # training as batches
    for batch_idx, batch_val in enumerate(range(0,total_length ,batch_size)):
        if batch_val+batch_size > total_length: break
        optimizer.zero_grad()  # zero the parameter gradients
        input_features = train_features_sfl[batch_val:batch_val+batch_size]
        input_val = train_val_sfl[batch_val:batch_val+batch_size]
        input_features = input_features.to(device)
        input_val = input_val.to(device)
        # forward + backward + optimize
        predict_labels,_ = model(input_features) #size 64x11
        loss = loss_function(predict_labels, input_val) #size 64x11 vs 64
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().data.numpy()
        num_batch = batch_idx+1
    print("avg training loss:",total_loss / num_batch)
    logfile.writelines("avg training loss:"+ str(total_loss / num_batch)+"\n")
    logfile.flush()
    train_loss.append(total_loss / num_batch)

    # validation
    with torch.no_grad():
        model.eval()
        predict_labels,_ = model(valid_features.to(device))
        predict_val = torch.argmax(predict_labels,1).cpu().data
        accuracy = np.mean((predict_val == valid_val).numpy())
        print("validation accuracy: ",accuracy)
        logfile.writelines("validation accuracy: "+str(accuracy)+"\n")
        logfile.flush()
        valid_acc.append(accuracy)

    # saving best acc model as best.pth
    if (accuracy > max_accuracy) :
        max_accuracy = accuracy
        torch.save(model, 'best_cnnp2.pth')
        logfile.writelines("save as best_cnnp2.pth\n")
        logfile.flush()
    model.train()

now = datetime.datetime.now()
logfile.writelines("end training at:"+str(now)+"\n")
logfile.flush()

# plot loss and acc graph
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(train_loss)
plt.title("training loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.subplot(1,2,2)
plt.plot(valid_acc)
plt.title("validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig("hw4_p1.png")
plt.show()