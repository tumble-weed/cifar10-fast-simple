import model
import torch.nn as nn

def get_model(arch, dataset, dtype, train_data=None):
    if(arch=='resnet8'):
        if(dataset in ['cifar10', 'mnist']):
            # Compute special weights for first layer
            weights = model.patch_whitening(train_data[:10000, :, 4:-4, 4:-4])

            # Construct the neural network
            train_model = model.Model(weights, c_in=3, c_out=10, scale_out=0.125)

            # Convert model weights to half precision
            train_model.to(dtype)

            # Convert BatchNorm back to single precision for better accuracy
            for module in train_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.float()

            return train_model
        elif(dataset=='cifar100'):
            dutils.TODO
        else: 
            dutils.TODO
    elif(arch=='vgg16'):
        if(dataset in ['cifar10', 'mnist']):
            from pytorch_vgg_cifar10 import vgg
            train_model = vgg.__dict__[arch]()
            train_model.to(dtype)
            # dutils.pause()
            return train_model
        elif(dataset=='cifar100'):
            dutils.TODO
    else:
        dutils.TODO