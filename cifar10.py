from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
#batch_size = 64
batch_size = 100


transform_train = transforms.Compose([transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                                      # Rotates the image to a specified angel
                                      transforms.RandomRotation(10),
                                      # Performs actions like zooms, change shear angles.
                                      transforms.RandomAffine(
                                          0, shear=10, scale=(0.8, 1.2)),
                                      # Set the color params
                                      transforms.ColorJitter(
                                          brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
                                      # Normalize all the images
                                      transforms.Normalize(
                                          (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])
train_dataset = datasets.CIFAR10(root='./data/',
                                 train=True,

                                 transform=transform_train,
                                 download=True)


transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])


test_dataset = datasets.CIFAR10(root='./data/',
                                train=False,
                                transform=transform_test)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        # We double the feature maps for every conv layer as in pratice it is really good.
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
        self.fc1 = nn.Linear(4*4*64, 500)
        self.dropout1 = nn.Dropout(0.5)
        # output nodes are 10 because our dataset have 10 different categories
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply relu to each output of conv layer.
        # Max pooling layer with kernal of 2 and stride of 2
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))

        x = F.max_pool2d(x, 2, 2)
        # print("Here: ", x.shape)  # Here:  torch.Size([100, 64, 4, 4])
        # flatten our images to 1D to input it to the fully connected layers
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        # Applying dropout b/t layers which exchange highest parameters. This is a good practice
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


model = Net()
# same as categorical_crossentropy loss used in Keras models which runs on Tensorflow
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # fine tuned the lr


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data, target = Variable(data, volatile=True), Variable(
            target)  # Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 15):
    train(epoch)
    test()
