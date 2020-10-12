import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from pynvml import *
import os

from models.net import net
os.environ['CUDA_VISIBLE_DEVICES']='0'

cuda = torch.cuda.is_available()
model = net()
if cuda:
    model = model.cuda()
model.eval()
model.load_state_dict(torch.load('model_0001_train_acc_0.9903109452736318_test_acc_0.9427083333333334.pth'))

refer = 'NFI-00101001_real1.png'
test = 'NFI-00101001_real2.png'

refer_img = cv2.imread(refer, 0)
refer_img = cv2.resize(refer_img, (220, 155), cv2.INTER_LINEAR)
refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])

test_img = cv2.imread(test, 0)
test_img = cv2.resize(test_img, (220, 155), cv2.INTER_LINEAR)
test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])

refer_test = np.array([np.concatenate((refer_img, test_img), axis=0)])
refer_test = torch.from_numpy(refer_test).float()

if cuda:
	refer_test = refer_test.cuda()

predicted = model(refer_test)
print(predicted)