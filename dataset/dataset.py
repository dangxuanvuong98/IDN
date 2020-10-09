from torch.utils import data
import torch
import cv2
import numpy as np

class dataset(data.Dataset):
    def __init__(self, root='CEDAR/', train=True):
        super(dataset, self).__init__()
        if train:
            path = root + 'train_data.csv'
            root = root + 'train/'
        else:
            path = root + 'test_data.csv'
            root = root + 'test/'
        
        with open(path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        for line in lines:
            refer, test, label = line.split(',')
            # print(root + refer)
            refer_img = cv2.imread(root + refer, 0)
            refer_img = cv2.resize(refer_img, (220, 155), cv2.INTER_LINEAR)
            test_img = cv2.imread(root + test, 0)
            test_img = cv2.resize(test_img, (220, 155), cv2.INTER_LINEAR)
            refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])
            test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])

            refer_test = np.concatenate((refer_img, test_img), axis=0)
            self.datas.append(refer_test)
            self.labels.append(int(label))

        # print(self.datas[0].shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.FloatTensor(self.datas[index]), float(self.labels[index])

# img = cv2.imread('dataset/original_2_9.png')
# print(img.shape)