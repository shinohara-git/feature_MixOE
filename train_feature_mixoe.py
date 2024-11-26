import os, argparse, time, json, sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trn
import torchvision.datasets as dset
from torchvision.models import resnet50
import torchvision.models as models

import matplotlib.pyplot as plt

from utils import (
    FinegrainedDataset, SPLIT_NUM_CLASSES, INET_SPLITS, WebVision,
    AverageMeter, accuracy , save_np , ResNet
)


class SoftCE(nn.Module):
    def __init__(self, reduction="mean"):
        super(SoftCE, self).__init__()
        self.reduction = reduction

    def forward(self, logits, soft_targets):
        preds = logits.log_softmax(dim=-1)
        # print(f"Predictions shape: {preds.shape}, Soft targets shape: {soft_targets.shape}")
        # soft_targets = soft_targets.view(preds.shape)

        assert preds.shape == soft_targets.shape

        loss = torch.sum(-soft_targets * preds, dim=-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError("Reduction type '{:s}' is not supported!".format(self.reduction))


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



# /////////////// Setup ///////////////
# Arguments
parser = argparse.ArgumentParser(description='Trains a classifier')
# Dataset options
parser.add_argument('--dataset', type=str, choices=['bird', 'butterfly', 'car', 'aircraft'],
                    help='Choose the dataset', required=True)
parser.add_argument('--data-dir', type=str, default='../../../dataset')
parser.add_argument('--info-dir', type=str, default='../info')
parser.add_argument('--split', '-s', type=int, default=0)
# Model options
parser.add_argument('--model', '-m', type=str, default='rn', help='Choose architecture.')
# MixOE options
parser.add_argument('--mix-op', type=str, choices=['mixup', 'cutmix'], required=True)
parser.add_argument('--oe-loss', action='store_true')
parser.add_argument('--feature_layer', type=str, choices=['conv1','layer1','layer2',"layer3",'layer4','fc'], required=True)
parser.add_argument('--alpha', type=float, default=1.0, help='Parameter for Beta distribution.')
parser.add_argument('--beta', type=float, default=1.0, help='Weighting factor for the OE objective.')
parser.add_argument('--oe-set', type=str, default='WebVision', choices=['WebVision'])
# Optimization options
parser.add_argument('--torch-seed', '-ts', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size.')
parser.add_argument('--test-bs', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.00001, help='Weight decay (L2 penalty).')
parser.add_argument('--add_noise', type=float, default=0.4, metavar='S', help='level of additive noise')
parser.add_argument('--mult_noise', type=float, default=0.2, metavar='S', help='level of multiplicative noise')
# Checkpoints
parser.add_argument('--save-dir', type=str, help='Folder to save checkpoints.')
# Acceleration
parser.add_argument('--gpu', nargs='*', type=int, default=[0,1])
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()


DATA_DIR = {
    'bird': os.path.join(args.data_dir, 'bird', 'images'),
    'butterfly': os.path.join(args.data_dir, 'butterfly', 'images_small'),
    'dog': os.path.join(args.data_dir, 'dog'),
    'car': os.path.join(args.data_dir, 'car'),
    'aircraft': os.path.join(args.data_dir, 'aircraft', 'images')
}

# Set random seed for torch
torch.manual_seed(args.torch_seed)

# Create ID/OOD splits
all_classes = list(range(sum(SPLIT_NUM_CLASSES[args.dataset])))
rng = np.random.RandomState(args.split)
id_classes = list(rng.choice(all_classes, SPLIT_NUM_CLASSES[args.dataset][0], replace=False))
ood_classes = [c for c in all_classes if c not in id_classes]
print(f'# ID classes: {len(id_classes)}, # OOD classes: {len(ood_classes)}')

# Datasets and dataloaders
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transform = trn.Compose([
    trn.Resize((512, 512)),
    trn.RandomCrop((448, 448)),
    trn.RandomHorizontalFlip(),
    trn.ColorJitter(brightness=32./255., saturation=0.5),
    trn.ToTensor(),
    trn.Normalize(mean, std),
])
test_transform = trn.Compose([
    trn.Resize((512, 512)),
    trn.CenterCrop(448),
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

train_set = FinegrainedDataset(
    image_dir=DATA_DIR[args.dataset], 
    info_filepath=os.path.join(args.info_dir, args.dataset, 'train.txt'), 
    class_list=id_classes,
    transform=train_transform
)
test_set = FinegrainedDataset(
    image_dir=DATA_DIR[args.dataset], 
    info_filepath=os.path.join(args.info_dir, args.dataset, 'val.txt'), 
    class_list=id_classes,
    transform=test_transform
)
num_classes = train_set.num_classes
assert num_classes == len(id_classes)
print(f'Train samples: {len(train_set)}, Val samples: {len(test_set)}')

if args.oe_set == 'WebVision':
    class_to_be_removed = []
    for l in INET_SPLITS.values():
        class_to_be_removed.extend(l)
    oe_set = WebVision(
        root=os.path.join(args.data_dir, 'WebVision'), transform=train_transform,
        concept_list=class_to_be_removed, exclude=True,
    )
else:
    raise NotImplementedError

train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, 
    num_workers=args.prefetch, pin_memory=False, drop_last=True
)
test_loader = DataLoader(
    test_set, batch_size=args.test_bs, shuffle=False, 
    num_workers=args.prefetch, pin_memory=False
)
oe_loader = DataLoader(
    oe_set, batch_size=args.batch_size, shuffle=True, 
    num_workers=args.prefetch, pin_memory=False
)

# Set up checkpoint directory and tensorboard writer
if args.save_dir is None:
    mixoe_related = f'feature_{args.mix_op}'
    mixoe_related += f'_{args.oe_set}_alpha={args.alpha:.1f}_beta={args.beta:.1f}'

    args.save_dir = os.path.join(
        '../checkpoints', args.dataset, 
        f'split_{args.split}',
        f'{args.model}_{mixoe_related}_epochs={args.epochs}_bs={args.batch_size}'
    )
else:
    assert 'checkpoints' in args.save_dir, \
        "If 'checkpoints' not in save_dir, then you may have an unexpected directory for writer..."
chkpnt_path = os.path.join(args.save_dir, f'seed_{args.torch_seed}.pth')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
elif os.path.isfile(chkpnt_path):
    print('*********************************')
    print('* The checkpoint already exists *')
    print('*********************************')

writer = SummaryWriter(args.save_dir.replace('checkpoints', 'runs'))

# Set up GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), args.gpu))

# Create model
if args.model == 'rn':
    net = ResNet(num_classes = num_classes)
    pretrained_model_file = f'../checkpoints/{args.dataset}/split_{args.split}/rn_baseline_epochs=90_bs=32_FMOE/seed_0.pth'
    state_dict = torch.load(pretrained_model_file,weights_only=True)
else:
    raise NotImplementedError

in_features = net.fc.in_features
new_fc = nn.Linear(in_features, num_classes)
net.fc = new_fc

# custom_state_dict = net.state_dict()
# mapped_state_dict = {}

# for key in state_dict.keys():
#     # 例: downsampleをidentity_downsampleに変換
#     new_key = key.replace("downsample", "identity_downsample")
    
#     # 自作モデルのキーに存在する場合のみマッピング
#     if new_key in custom_state_dict:
#         mapped_state_dict[new_key] = state_dict[key]

# custom_state_dict.update(mapped_state_dict)
# net.load_state_dict(custom_state_dict)

try:
    net.load_state_dict(state_dict,strict=False)
except RuntimeError:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.` caused by nn.DataParallel
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict,strict=False)
# print(net)
net.cuda()
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
cudnn.benchmark = True  # fire on all cylinders

# Optimizer and scheduler
optimizer = optim.SGD(
    net.parameters(), args.lr, momentum=args.momentum,
    weight_decay=args.decay, nesterov=True
)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.lr)
)

soft_xent = SoftCE()
    

# /////////////// Training ///////////////
# train function
def train():
    net.train()  # enter train mode

    current_lr = scheduler.get_last_lr()[0]
    id_losses = AverageMeter('ID_Loss', ':.4e')
    if args.oe_loss:
        oe_losses = AverageMeter('OE_Loss', ':.4e')
    mixed_losses = AverageMeter('Mixed_Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    batch_iter = tqdm(train_loader, total=len(train_loader), 
        desc='Batch', leave=False, position=2)

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    oe_loader.dataset.offset = np.random.randint(len(oe_loader.dataset))
    oe_iter = iter(oe_loader)
    

    for x, y in batch_iter:
        bs = x.size(0)
        try:
            oe_x, _ = next(oe_iter)
        except StopIteration:
            continue
        assert bs == oe_x.size(0)

        x, y = x.cuda(), y.cuda()
        oe_x = oe_x.cuda()
        one_hot_y = torch.zeros(bs, num_classes).cuda()
        one_hot_y.scatter_(1, y.view(-1, 1), 1)
        oe_y = torch.ones(oe_x.size(0), num_classes).cuda() / num_classes

        lam = np.random.beta(args.alpha, args.alpha)

        # ID Loss
        logits = net(x)
        id_loss = F.cross_entropy(logits, y)
        # OE Loss
        if args.oe_loss:
            logits = net(oe_x)
            oe_loss = F.cross_entropy(logits, oe_y)

        if args.mix_op == 'cutmix':
            mixed_x = x.clone().detach()
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # we empirically find that pasting outlier patch into ID data performs better
            # than pasting ID patch into outlier data
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = oe_x[:, :, bbx1:bbx2, bby1:bby2]  
        elif args.mix_op == 'mixup':
            soft_labels = lam * one_hot_y + (1 - lam) * oe_y
            x_cat = torch.cat((x, oe_x), dim=0)
            mixed_loss = soft_xent(net(x_cat,lam,args,mix = True), soft_labels)
        

        if args.oe_loss :
            loss = id_loss + oe_loss + args.beta * mixed_loss
        else :
            loss = id_loss + args.beta * mixed_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1, acc5 = accuracy(net(x), y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        id_losses.update(id_loss.item(), x.size(0))
        if args.oe_loss:
            oe_losses.update(oe_loss.item(), x.size(0))
        mixed_losses.update(mixed_loss.item(), x.size(0))
        top1.update(acc1, x.size(0))
        top5.update(acc5, x.size(0))
    
    print_message = f'Epoch [{epoch:3d}] | ID Loss: {losses.avg:.4f}, ' \
        f'Top1 Acc: {top1.avg:.2f}, Top5 Acc: {top5.avg:.2f}'
    tqdm.write(print_message)

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/acc_top1', top1.avg, epoch)
    writer.add_scalar('train/acc_top5', top5.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    state['train_loss'].append(losses.avg)
    state['train_accuracy'].append(top1.avg)
    loss_log['id_loss'].append(id_losses.avg)
    if args.oe_loss:   
        loss_log['oe_loss'].append(oe_losses.avg)
    loss_log['mixed_loss'].append(mixed_losses.avg)


# test function
def test():
    net.eval()

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    loss_avg = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            output = net(x)

            loss = F.cross_entropy(output , y)

            loss_avg += float(loss.data)
            
            acc1, acc5 = accuracy(output, y, topk=(1, 5))
            top1.update(acc1, x.size(0))
            top5.update(acc5, x.size(0))
    
    print_message = f'Evaluation  | Top1 Acc: {top1.avg:.2f}, Top5 Acc: {top5.avg:.2f}\n'
    tqdm.write(print_message)

    writer.add_scalar('test/acc_top1', top1.avg, epoch)
    writer.add_scalar('test/acc_top5', top5.avg, epoch)
    state['test_accuracy'].append(top1.avg)
    state['test_loss'].append(loss_avg / len(test_loader))

    return top1.avg

# Main loop
epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',
    leave=True, position=1)

state = {'train_loss':[],'train_accuracy':[],'test_loss':[],'test_accuracy':[]}
loss_log = {'id_loss':[],'oe_loss':[],'mixed_loss':[]}
best_acc1 = 0
if args.oe_loss:
    path = "../savefig/feature_mixoe/{}/{}/{}/with_oe_loss/split_{}".format(args.mix_op,args.feature_layer,args.dataset,args.split)
else :
    path = "../savefig/feature_mixoe/{}/{}/{}/without_oe_loss/split_{}".format(args.mix_op,args.feature_layer,args.dataset,args.split)
os.makedirs(path, exist_ok=True)
for epoch in epoch_iter:
    train()
    acc1 = test()

    if acc1 > best_acc1:
        # Save model
        torch.save(
            net.state_dict(),
            chkpnt_path
        )

    best_acc1 = max(acc1, best_acc1)

    plt.clf()
    plt.plot(state["train_loss"], label="train")
    plt.plot(state["test_loss"], label="test")
    plt.plot(loss_log["id_loss"], label="id_loss")
    plt.plot(loss_log["mixed_loss"], label="mixed_loss")
    if args.oe_loss:
        plt.plot(loss_log["oe_loss"], label="oe_loss")
    plt.title("train and test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.savefig(path + "/losses.png")

    keys = ['train_loss','test_loss']
    sub_dict = {key: state[key] for key in keys if key in state}
    save_np("/loss",nump=sub_dict,path=path)

    plt.clf()
    plt.plot(state["train_accuracy"], label="train")
    plt.plot(state["test_accuracy"], label="test")
    plt.title("train and test accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(path + "/accuracy.png")

    keys = ['train_accuracy','test_accuracy']
    sub_dict = {key: state[key] for key in keys if key in state}
    save_np("/accuracy",nump=sub_dict,path=path)
