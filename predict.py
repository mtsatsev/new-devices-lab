import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_file = "data/database.csv"
root_dir = "data/"
batch_size = 200
shuffle = True


class Data(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.anotaion = pd.read_csv(csv_file)
        self.anotaion = sorted(self.anotaion, key=lambda x: x[-1])
        self.anotaion = __padding__(anotaion)
        self.anotaion = self.anotaion.transpose()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.anotaion) #20 000 images 2000 each

    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir, self.anotaion.iloc[index,0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSC)
        y_label = torch.tensor(int(self.anotaion.iloc[index,1]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)

    def __padding__(data):
        return np.pad(data,160,pad_with)


class Model(nn.Module):
    def __init__(self,input_size, hidden_size,n_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size)
        self.conv2 = nn.Conv2d()

        self.l1 = nn.Linear()


    def forward(self, x):
        pass
    def predict(image):
        x = self.forward(image)
        return np.argmax(x)

    def train(self,train_data,valid_data,learning_rate,num_epochs,optimizer,criterion):
        train_loss = np.zeros(num_epochs)
        valid_loss = np.zerps(num_epochs)
        train_accuracy = np.zeros(num_epochs)
        valid_accuracy = np.zeros(num_epochs)

        #begin training
        for epoch in range(num_epochs):
            train_losses = []
            train_correct= 0
            total_items  = 0

            valid_losses = []
            valid_correct = 0

            for images,labels in train_data:
                optimizer.zero_grad()

                #add to GPU hopefully
                images = images.to(device)
                labels = labels.to(device)

                #Forward pass
                outputs = self.forward(images)
                loss    = criterion(outputs,labels)

                #Backward pass
                loss.backward()
                optimizer.step()

                #staticstics
                train_losses.append(loss.item())
                _, predicted = torch.max(outputs.data,1)
                train_correct += (predicted == labels).sum().item()
                total_items += labels.size(0)

            train_loss[epoch] = np.mean(train_losses)
            train_accuracy[epoch] = (100 * train_correct/total_items)

            with torch.no_grad():
                correct_val = 0
                total_val = 0

                for images,labels in valid_data:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = self.forward(images)
                    loss    = criterion(outputs, labels)

                    valid_losses.append(loss.item())
                    _, predicted = torch.max(outputs.data, 1)

                    correct += (predicted == labels).sum().item()
                    total   += labels.size(0)

            valid_loss[epoch] = np.mean(valid_losses)
            valid_accuracy[epoch] = (100 * correct/total)

            print("Epoch: [{},{}], train accuracy: {}, valid accuracy: {}, train loss: {}, valid loss: {}"
            .format(num_epochs,epoch,train_accuracy[epoch],valid_accuracy[epoch],train_loss[epoch],valid_accuracy[epoch]))

        return valid_loss, valid_accuracy,train_loss,train_accuracy

    def test(self,test_data,test_labels):
        outputs = []
        targets = []
        with torch.no_grad():
            for image,label in test_data:
                images = image.to(device)
                labels = labels.to(device)

                prediction = elf.predict(images)
                outputs.append(prediction)
                targets.append(labels)

        return targets, outputs


def Dataset(csv_file,root,batch_size,shuffle,):
    dataset = Data(csv_file, root_dir=root,transform=transforms.ToTensor())
    print(dataset.__len__())
    train, test = torch.utils.data.random_split(dataset, [12000,8000])
    valid, test = torch.utils.data.random_split(test, [5000,3000])

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=shuffle)

    test_loader  = DataLoader(dataset=test, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader



def pad_with(vector,pad_width,iaxis,kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def main():
    print("Start")
    train,valid,test = Dataset(csv_file, root_dir, batch_size, shuffle)
    print("HEYLYA")

main()
