from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data = datasets.MNIST(
    root='data',
    train = True,
    transform= ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root='data',
    train = False,
    transform= ToTensor(),
    download=True
)

inputs = train_data.data.shape  # 60_000 x 28 x 28
targets = train_data.targets    # 60_000 (what each matrix number is)

loaders = {
    'train': DataLoader(train_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=1),
    'test': DataLoader(test_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=1)
}

