import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import model
import copy
from torch.utils.data import DataLoader
from resnet8_fast_simple.scheduler import get_scheduler
from resnet8_fast_simple.optimizer import get_optimizer
from resnet8_fast_simple.get_model import get_model
from resnet8_fast_simple.dataset import get_dataset
import dutils
dutils.init()
 
def seed_everything(SEED = 0):
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

def random_crop(data, crop_size):
    crop_h, crop_w = crop_size
    h = data.size(2)
    w = data.size(3)
    x = torch.randint(w - crop_w, size=(1,))[0]
    y = torch.randint(h - crop_h, size=(1,))[0]
    return data[:, :, y : y + crop_h, x : x + crop_w]

def train(args):
    if args.device != 'cpu':
        print("Device :", torch.cuda.get_device_name(args.device.index))
    
    seed_everything(args.seed)

    train_data, train_targets, valid_data, valid_targets = get_dataset(
        arch = args.arch, dataset = args.dataset, device = args.device, data_dir = args.data_dir, dtype = args.dtype)
    
    train_model = get_model(arch = args.arch, dataset = args.dataset, dtype = args.dtype, train_data = train_data)
    
    train_model.to(args.device)

    valid_model = dutils.hardcode(valid_model = train_model)

    lr_scheduler = get_scheduler(args.lr_schedule)

    optimizer = get_optimizer(
        args.optimizer, weight_decay=args.weight_decay, weight_decay_bias=args.weight_decay_bias,
        momentum=args.momentum, ema_update_freq=args.ema_update_freq, rho=args.ema_rho)

    # Collect weights and biases and create nesterov velocity values
    weights = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) > 1
    ]
    biases = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) <= 1
    ]

    # Copy the model for validation
    valid_model = copy.deepcopy(train_model)
    batch_count = 0

    for epoch in range(1, args.epochs+1):
        indices = torch.randperm(len(train_data), device=args.device)
        data = train_data[indices]
        targets = train_targets[indices]

        data = [
            random_crop(data[i : i + args.batch_size], crop_size=(32, 32))
            for i in range(0, len(data), args.batch_size)
        ]
        data = torch.cat(data)

        # Randomly flip half the training data
        data[: len(data) // 2] = torch.flip(data[: len(data) // 2], [-1])

        for i in range(0, len(data), args.batch_size):
            # discard partial batches
            if i + args.batch_size > len(data):
                break

            # Slice batch from data
            inputs = data[i : i + args.batch_size]
            target = targets[i : i + args.batch_size]
            batch_count += 1

            # Compute new gradients
            train_model.zero_grad()
            train_model.train(True)

            logits = train_model(inputs)

            loss = model.label_smoothing_loss(logits, target, alpha=0.2)
            # loss = criterion(logits, target)

            loss.sum().backward()

            optimizer.step(weights, biases, lr_scheduler, train_model, valid_model)
            lr_scheduler.step()

        valid_correct = []
        for i in range(0, len(valid_data), args.batch_size):
            valid_model.train(False)

            # Test time agumentation: Test model on regular and flipped data
            regular_inputs = valid_data[i : i + args.batch_size]
            flipped_inputs = torch.flip(regular_inputs, [-1])

            logits1 = valid_model(regular_inputs).detach()
            logits2 = valid_model(flipped_inputs).detach()

            # Final logits are average of augmented logits
            logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)

            # Compute correct predictions
            correct = logits.max(dim=1)[1] == valid_targets[i : i + args.batch_size]

            valid_correct.append(correct.detach().type(torch.float64))

        # Accuracy is average number of correct predictions
        valid_acc = torch.mean(torch.cat(valid_correct)).item()

        print(f"{epoch:5} {batch_count:8d} {valid_acc:22.4f}")
    
    dutils.pause()
    mypath = os.path.abspath(__file__)
    mydir = os.path.dirname(mypath)
    savepath = os.path.join(mydir,args.save_dir,'test_checkpoint.pth')
    torch.save(valid_model.state_dict(),savepath)
    return valid_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', metavar='ARCH', default='resnet8', help='model architecture')
    parser.add_argument('--dataset', default='cifar10', help='dataset: cifar10, cifar100, mnist')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default:4')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.256, type=float)
    parser.add_argument('--weight_decay_bias', default=0.004, type=float)
    parser.add_argument('--half', action='store_true', help='use half-precision(16-bit)')
    parser.add_argument('--save_dir', help='The directory used to save the trained model', default='save_temp', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_dir', default='"~/data')
    parser.add_argument('--dtype', default=torch.float16)
    parser.add_argument('--lr_schedule', default='custom_scheduler1', type=str)
    parser.add_argument('--optimizer', default='nesterov', type=str)
    parser.add_argument('--ema_update_freq', default=5, type=int)
    # parser.add_argument('--ema_rho', default=0.99 ** parser.ema_update_freq, type=float)
    
    global args
    args = parser.parse_args()
    args.ema_rho = 0.99 ** args.ema_update_freq
    if args.dtype == None:
        args.dtype = torch.float32
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train(args) 

if __name__ == "__main__":
    main()
