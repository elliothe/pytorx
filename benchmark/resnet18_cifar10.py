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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, crxb_cfg, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = crxb_Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False, **crxb_cfg)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = crxb_Conv2d(planes, planes, kernel_size=3, stride=1,
                                 padding=1, bias=False, **crxb_cfg)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                crxb_Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                            bias=False, **crxb_cfg),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10/100 mapped onto ReRAM crossbar arrays.

    Uses 3x3 initial conv (no maxpool) suitable for 32x32 inputs.
    """

    def __init__(self, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop,
                 freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF, num_classes=10,
                 quantize=8, weight_precision=None):
        super(ResNet18, self).__init__()

        crxb_cfg = dict(crxb_size=crxb_size, scaler_dw=scaler_dw,
                        gwire=gwire, gload=gload, gmax=gmax, gmin=gmin,
                        vdd=vdd, freq=freq, temp=temp,
                        enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                        enable_noise=enable_noise, ir_drop=ir_drop,
                        device=device, quantize=quantize,
                        weight_precision=weight_precision)

        self.in_planes = 64

        self.conv1 = crxb_Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                 bias=False, **crxb_cfg)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1, crxb_cfg=crxb_cfg)
        self.layer2 = self._make_layer(128, 2, stride=2, crxb_cfg=crxb_cfg)
        self.layer3 = self._make_layer(256, 2, stride=2, crxb_cfg=crxb_cfg)
        self.layer4 = self._make_layer(512, 2, stride=2, crxb_cfg=crxb_cfg)
        self.fc = crxb_Linear(512, num_classes, **crxb_cfg)

    def _make_layer(self, planes, num_blocks, stride, crxb_cfg):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, crxb_cfg, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


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
    parser = argparse.ArgumentParser(description='PyTorX ResNet18 CIFAR-10 Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
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
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False,
                         transform=transform_test),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = ResNet18(crxb_size=args.crxb_size, gmax=args.gmax, gmin=args.gmin,
                     gwire=args.gwire, gload=args.gload, vdd=args.vdd,
                     ir_drop=args.ir_drop, device=device, scaler_dw=args.scaler_dw,
                     freq=args.freq, temp=args.temp, enable_SAF=args.enable_SAF,
                     enable_noise=args.enable_noise,
                     enable_ec_SAF=args.enable_ec_SAF,
                     num_classes=10, quantize=args.quantize,
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

            filename = 'checkpoint_resnet18_cifar10_' + str(args.crxb_size) + '.pth.tar'
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_error,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=filename)

    elif args.test:
        modelfile = 'checkpoint_resnet18_cifar10_' + str(args.crxb_size) + '.pth.tar'
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
