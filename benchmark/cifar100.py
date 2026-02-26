from __future__ import print_function

import argparse
import os
import shutil
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from python.torx.module.layer import crxb_Conv2d
from python.torx.module.layer import crxb_Linear


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VGG9(nn.Module):
    """VGG9 network for CIFAR-100 mapped onto ReRAM crossbar arrays.

    Architecture:
        Conv 128 -> Conv 128 -> MaxPool ->
        Conv 256 -> Conv 256 -> MaxPool ->
        Conv 512 -> Conv 512 -> MaxPool ->
        FC 1024 -> FC 1024 -> FC 100
    """

    def __init__(self, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop,
                 freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF, num_classes=100,
                 quantize=8, weight_precision=None):
        super(VGG9, self).__init__()

        crxb_cfg = dict(crxb_size=crxb_size, scaler_dw=scaler_dw,
                        gwire=gwire, gload=gload, gmax=gmax, gmin=gmin,
                        vdd=vdd, freq=freq, temp=temp,
                        enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                        enable_noise=enable_noise, ir_drop=ir_drop,
                        device=device, quantize=quantize,
                        weight_precision=weight_precision)

        # Block 1
        self.conv1 = crxb_Conv2d(3, 128, kernel_size=3, padding=1, **crxb_cfg)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = crxb_Conv2d(128, 128, kernel_size=3, padding=1, **crxb_cfg)
        self.bn2 = nn.BatchNorm2d(128)

        # Block 2
        self.conv3 = crxb_Conv2d(128, 256, kernel_size=3, padding=1, **crxb_cfg)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = crxb_Conv2d(256, 256, kernel_size=3, padding=1, **crxb_cfg)
        self.bn4 = nn.BatchNorm2d(256)

        # Block 3
        self.conv5 = crxb_Conv2d(256, 512, kernel_size=3, padding=1, **crxb_cfg)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = crxb_Conv2d(512, 512, kernel_size=3, padding=1, **crxb_cfg)
        self.bn6 = nn.BatchNorm2d(512)

        # Classifier
        self.fc1 = crxb_Linear(512 * 4 * 4, 1024, **crxb_cfg)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = crxb_Linear(1024, 1024, **crxb_cfg)
        self.bn8 = nn.BatchNorm1d(1024)
        self.fc3 = crxb_Linear(1024, num_classes, **crxb_cfg)

    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)

        # Block 2: 16x16 -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)

        # Block 3: 8x8 -> 4x4
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(F.relu(self.bn6(self.conv6(x))), 2)

        # Classifier
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.bn7(self.fc1(x))), 0.5, training=self.training)
        x = F.dropout(F.relu(self.bn8(self.fc2(x))), 0.5, training=self.training)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


def train(model, device, criterion, optimizer, train_loader, epoch):
    losses = AverageMeter()

    model.train()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        for name, module in model.named_modules():
            if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
                module._reset_delta()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_loader.sampler.__len__(),
                       100. * batch_idx / len(train_loader), loss.item()))

    print('\nTrain set: Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, train_loader.sampler.__len__(),
        100. * correct / train_loader.sampler.__len__()))

    return losses.avg


def validate(args, model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            if args.ir_drop:
                print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
                    correct, val_loader.batch_sampler.__dict__['batch_size'] * (batch_idx + 1),
                             100. * correct / (val_loader.batch_sampler.__dict__['batch_size'] * (batch_idx + 1))))

        test_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, val_loader.sampler.__len__(),
            100. * correct / val_loader.sampler.__len__()))

        return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorX VGG9 CIFAR-100 Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--crxb_size', type=int, default=64, help='crossbar size')
    parser.add_argument('--vdd', type=float, default=3.3, help='supply voltage')
    parser.add_argument('--gwire', type=float, default=0.0357,
                        help='wire conductance')
    parser.add_argument('--gload', type=float, default=0.25,
                        help='load conductance')
    parser.add_argument('--gmax', type=float, default=0.000333,
                        help='maximum cell conductance')
    parser.add_argument('--gmin', type=float, default=0.000000333,
                        help='minimum cell conductance')
    parser.add_argument('--ir_drop', action='store_true', default=False,
                        help='switch to turn on ir drop analysis')
    parser.add_argument('--scaler_dw', type=float, default=1,
                        help='scaler to compress the conductance')
    parser.add_argument('--test', action='store_true', default=False,
                        help='switch to turn on inference mode')
    parser.add_argument('--enable_noise', action='store_true', default=False,
                        help='switch to turn on noise analysis')
    parser.add_argument('--enable_SAF', action='store_true', default=False,
                        help='switch to turn on SAF analysis')
    parser.add_argument('--enable_ec_SAF', action='store_true', default=False,
                        help='switch to turn on SAF error correction')
    parser.add_argument('--freq', type=float, default=10e6,
                        help='operating frequency')
    parser.add_argument('--temp', type=float, default=300,
                        help='operating temperature')
    parser.add_argument('--quantize', type=int, default=8,
                        help='quantization resolution of the crossbar')
    parser.add_argument('--weight_precision', type=int, default=None,
                        help='bit precision for ReRAM conductance levels (default: same as quantize)')

    args = parser.parse_args()

    best_error = float('inf')

    if args.ir_drop and (not args.test):
        warnings.warn("We don't recommend training with IR drop, too slow!")

    if args.ir_drop and args.test_batch_size > 150:
        warnings.warn("Reduce the batch size, IR drop is memory hungry!")

    if not args.test and args.enable_noise:
        raise KeyError("Noise can cause unsuccessful training!")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True,
                          transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False,
                          transform=transform_test),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = VGG9(crxb_size=args.crxb_size, gmax=args.gmax, gmin=args.gmin,
                 gwire=args.gwire, gload=args.gload, vdd=args.vdd,
                 ir_drop=args.ir_drop, device=device, scaler_dw=args.scaler_dw,
                 freq=args.freq, temp=args.temp, enable_SAF=args.enable_SAF,
                 enable_noise=args.enable_noise,
                 enable_ec_SAF=args.enable_ec_SAF,
                 num_classes=100, quantize=args.quantize,
                 weight_precision=args.weight_precision).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs)

    loss_log = []
    if not args.test:
        for epoch in range(args.epochs):
            print("epoch {0}, and now lr = {1:.6f}\n".format(
                epoch, optimizer.param_groups[0]['lr']))
            train_loss = train(model=model, device=device, criterion=criterion,
                               optimizer=optimizer, train_loader=train_loader,
                               epoch=epoch)
            val_loss = validate(args=args, model=model, device=device,
                                criterion=criterion, val_loader=test_loader)

            scheduler.step()

            loss_log += [(epoch, train_loss, val_loss)]
            is_best = val_loss < best_error
            best_error = min(val_loss, best_error)

            filename = 'checkpoint_cifar100_' + str(args.crxb_size) + '.pth.tar'
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_error,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=filename)

    elif args.test:
        modelfile = 'checkpoint_cifar100_' + str(args.crxb_size) + '.pth.tar'
        if os.path.isfile(modelfile):
            print("=> loading checkpoint '{}'".format(modelfile))
            checkpoint = torch.load(modelfile)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(modelfile))

            validate(args=args, model=model, device=device,
                     criterion=criterion, val_loader=test_loader)


if __name__ == '__main__':
    main()
