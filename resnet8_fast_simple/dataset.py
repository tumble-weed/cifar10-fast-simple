import torch
import torchvision
import torch.nn as nn

def get_dataset(arch, dataset, device, data_dir='~/data', dtype=torch.float32):
    if(arch=='resnet8'):
        if(dataset=='cifar10'):
            return load_cifar10(device, dtype=dtype, valid_dtype=dtype)
        else:
            dutils.TODO
    elif(arch=='vgg16'):
        if(dataset=='cifar10'):
            return load_cifar10(device, dtype=dtype, valid_dtype=dtype)
    
    else:
        dutils.TODO

'''
def load_cifar10(device, dtype,valid_dtype=None, data_dir="~/data"):
    if valid_dtype is None:
        valid_dtype = dtype
    train = torchvision.datasets.CIFAR10(root=data_dir, download=True)
    valid = torchvision.datasets.CIFAR10(root=data_dir, train=False)

    train_data = preprocess_data_cifar(train.data, device, dtype)
    valid_data = preprocess_data_cifar(valid.data, device, valid_dtype)

    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)

    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)

    return train_data, train_targets, valid_data, valid_targets


def preprocess_data_cifar(data, device, dtype):
    # Convert to torch float16 tensor
    data = torch.tensor(data, device=device).to(dtype)

    # Normalize
    mean = torch.tensor([125.31, 122.95, 113.87], device=device).to(dtype)
    std = torch.tensor([62.99, 62.09, 66.70], device=device).to(dtype)
    data = (data - mean) / std

    # Permute data from NHWC to NCHW format
    data = data.permute(0, 3, 1, 2)

    return data
'''
def load_mnist(device, dtype,valid_dtype=None, data_dir="~/data"):
    if valid_dtype is None:
        valid_dtype = dtype
    train = torchvision.datasets.MNIST(root=data_dir, download=True)
    valid = torchvision.datasets.MNIST(root=data_dir, train=False)
    train_data = preprocess_data_mnist(train.data, device, dtype)
    valid_data = preprocess_data_mnist(valid.data, device, valid_dtype)

    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)

    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)

    return train_data, train_targets, valid_data, valid_targets

'''
def load_cifar10(device, dtype,valid_dtype=None, data_dir="~/data"):
    if valid_dtype is None:
        valid_dtype = dtype
    train = torchvision.datasets.CIFAR10(root=data_dir, download=True)
    valid = torchvision.datasets.CIFAR10(root=data_dir, train=False)

    train_data = preprocess_data_cifar(train.data, device, dtype)
    valid_data = preprocess_data_cifar(valid.data, device, valid_dtype)

    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)

    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)

    return train_data, train_targets, valid_data, valid_targets
'''

def load_cifar(device, dtype,valid_dtype=None,n_classes=None, data_dir="~/data"):
    if valid_dtype is None:
        valid_dtype = dtype
    if n_classes == 10:
        train = torchvision.datasets.CIFAR10(root=data_dir, download=True)
        valid = torchvision.datasets.CIFAR10(root=data_dir, train=False)
    elif n_classes == 100:
        train = torchvision.datasets.CIFAR100(root=data_dir, download=True)
        valid = torchvision.datasets.CIFAR100(root=data_dir, train=False)
    else:
        dutils.pause()

    train_data = preprocess_data_cifar(train.data, device, dtype)
    valid_data = preprocess_data_cifar(valid.data, device, valid_dtype)

    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)

    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)

    return train_data, train_targets, valid_data, valid_targets
def load_dataset(dataset,device,dtype,valid_dtype=None):
    if dataset == 'cifar-10':
        return load_cifar(device, dtype,valid_dtype=valid_dtype,n_classes=10)
    elif dataset == 'cifar-100':
        return load_cifar(device, dtype,valid_dtype=valid_dtype,n_classes=100)
    elif dataset == 'mnist':
        return load_mnist(device, dtype,valid_dtype=valid_dtype)
    else:
        dutils.pause()



