
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context




dataset_train = datasets.CIFAR10(root='data',
                                 train = True,
                                 transform= ToTensor(),
                                 download=True
)

dataset_val = datasets.CIFAR10(
                                 root='data',
                                 train = False,
                                 transform= ToTensor(),
                                 download=True
)

img_size = 32


preprocess1 = transforms.Compose([transforms.Resize((img_size, img_size)),
             transforms.ToTensor()
             ])

inputs_train = []
def train_inputs():
    for record in tqdm(dataset_train):
        image = record[0]
        label = record[1]

        if isinstance(image, torch.Tensor):
        # If image is already a tensor, convert it to PIL Image
            image = transforms.functional.to_pil_image(image)


        if image.mode == 'L':
            image = image.convert("RGB")
            print('hit')

        input_tensor = preprocess1(image)
        inputs_train.append([input_tensor, label])

train_inputs()


idx = np.random.randint(0, len(inputs_train), (512,))
tensors = torch.concat([inputs_train[i][0] for i in idx], axis=1)
tensors = tensors.view(-1, 3)
mean = torch.mean(tensors, 0)
std = torch.std(tensors, 0)



preprocess2 = transforms.Compose([transforms.Normalize(mean, std)])

preprocess3 = transforms.Compose([transforms.Resize((img_size, img_size)),
             transforms.ToTensor(), transforms.Normalize(mean, std)])




# idx = np.random.randint(0, len(inputs_train), (512,))

# tensors = torch.concat([inputs_train[i][0] for i in idx], axis=1)
# tensors.shape



inputs_val = []
def train_vals():
    for record in tqdm(dataset_train):
        image = record[0]
        label = record[1]

        if isinstance(image, torch.Tensor):
        # If image is already a tensor, convert it to PIL Image
            image = transforms.functional.to_pil_image(image)


        if image.mode == 'L':
            image = image.convert("RGB")
            print('hit')

        input_tensor = preprocess3(image)
        inputs_val.append([input_tensor, label])
train_vals()

batch_size = 64
dloader_train = torch.utils.data.DataLoader(inputs_train, batch_size, shuffle=True)
dloader_val = torch.utils.data.DataLoader(inputs_val, batch_size, shuffle=True)
num_classes = 10

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 4, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 4)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 4)
        self.pool2 = nn.MaxPool2d(kernel_size = 4, padding = 2)
        
        self.ln1 = nn.Linear(256, 256)
        self.ln2 = nn.Linear(256, 64)
        self.ln3 = nn.Linear(64, num_classes)
        
#         self.seq = nn.Sequential(
#             self.conv1,
#             nn.ReLU(),
#             self.pool1,
        
#             self.conv2,
#             nn.ReLU(),
#             self.pool2,
#             nn.Dropout(),
            
#             nn.Flatten(),
        
#             self.ln1,
#             self.ln2,
#         )
        
    def forward(self, x):
#         print("start", x.shape)
        x = self.conv1(x)
#         print('After conv1:', x.shape)
#         x = nn.ReLU()(x)
        x = self.pool1(x)
        x = nn.ReLU()(x)
#         print('After pool1:', x.shape)

        x = self.conv2(x)
#         print('After conv2:', x.shape)
        
#         x = nn.ReLU()(x)
        x = self.pool2(x)
#         print('After pool2:', x.shape)
        x = nn.ReLU()(x)
#         x = nn.Dropout()(x)
        x = nn.Flatten()(x)
#         print('After flatten:', x.shape)

        x = self.ln1(x)
#         print('After linear 1:', x.shape)
        x = nn.ReLU()(x)

        x = self.ln2(x)
        x = nn.ReLU()(x)
#         print('After linear 2:', x.shape)

        x = self.ln3(x)
        x = nn.ReLU()(x)
        
        x = F.softmax(x, dim=1)

        return x

model = Conv()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train():
    for batch_idx, (data, target) in enumerate(dloader_train):

        model.train()
        
        ypred = model(data)
        optimizer.zero_grad()
        loss = loss_fn(ypred, target)
        if batch_idx % 100 == 0:
            print(loss)
        loss.backward()
        optimizer.step()
    print('finished training')
        
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dloader_val):
            
            model.eval()
            ypred = model(data)
            test_loss += loss_fn(ypred, target)
            prediction = torch.argmax(ypred,dim=1)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            if batch_idx % 100 == 0:
                print('testing ...')

    print("test done")
    print(correct)

# print(f"percent: {correct / tested}")

def evaluate():
    torch.manual_seed(4)
    model.eval()
    idx = torch.randint(0, len(dataset_val) - 1, (1,))
    data = dataset_val[idx][0]
    data = data.unsqueeze(0)
    print(data.shape)
    target = dataset_val[idx][1]
    
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    
    image = data.squeeze(0, 1)
    print(image.shape)
    print(image[0].shape)
    print("pred", pred)
    print("target", target)
    plt.imshow(image[0])
    plt.show()


train()
evaluate()



