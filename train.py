import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynvml import *
import os
from tensorboardX import SummaryWriter
import time

from dataset.dataset import dataset
from models.net import net
from loss import loss
os.environ['CUDA_VISIBLE_DEVICES']='1'

def compute_accuracy(predicted, labels):
    for i in range(3):
        predicted[i][predicted[i] > 0.5] = 1
        predicted[i][predicted[i] <= 0.5] = 0
    predicted = predicted[0] + predicted[1] + predicted[2]
    
    predicted[predicted < 2] = 0
    predicted[predicted >= 2] = 1
    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy


BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001

np.random.seed(0)
torch.manual_seed(1)

cuda = torch.cuda.is_available()

train_set = dataset(root='./kaggle_data/', train=True)
test_set = dataset(root='./kaggle_data/', train=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2*BATCH_SIZE, shuffle=False)

model = net()
if cuda:
    model = model.cuda()
criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter(log_dir='scalar')

if cuda:
    criterion = criterion.cuda()
iter_n = 0
t = time.strftime("%m-%d-%H-%M", time.localtime())
print(len(train_loader))
for epoch in range(1, EPOCHS + 1):
    for i, (inputs, labels) in enumerate(train_loader):
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        labels = labels.float()
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        predicted = model(inputs)

        loss = criterion(*predicted, labels)  
        
        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(predicted, labels)

        writer.add_scalar(t+'/train_loss', loss.item(), iter_n)
        writer.add_scalar(t+'/train_accuracy', accuracy, iter_n)
        
        iter_n += 1

        if i == 500:
            torch.save(model.state_dict(), 'model.pth')

        print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch, EPOCHS, i, loss.item(), accuracy))

writer.close()
