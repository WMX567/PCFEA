import numpy as np
import random
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from utils.pc_utils import random_rotate_one_axis
import sklearn.metrics as metrics
import argparse
import json
import utils.log
from data.dataloader_PointDA_initial import ScanNet, ModelNet, ShapeNet, label_to_idx
from models.model import DGCNN

NWORKERS=4
MAX_LOSS = 9 * (10**9)
spl_weight = 1
cls_weight = 1


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--dataroot', type=str, default='../data/', metavar='N', help='data path')
parser.add_argument('--out_path', type=str, default='./pcfea/exp/', help='log folder path')
parser.add_argument('--exp_name', type=str, default='test2', help='Name of the experiment')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=10, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--cls_weight', type=float, default=0.8, help='weight of the classification loss')
parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for selecting target data')
parser.add_argument('--model_type', type = str, default='SPST', help = 'model name')
parser.add_argument('--type_', type = str, default='save_best_by_trgt_test', help = 'model type')
args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))
random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    io.cprint('Using GPUs')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


# ==================
# Utils
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


class DataLoad(Dataset):
    def __init__(self, io, data, partition='train'):
        self.partition = partition
        self.pc, self.label = data
        self.num_examples = len(self.pc)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in trgt_dataset : " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in trgt_dataset " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.pc[item])
        pointcloud = random_rotate_one_axis(pointcloud.transpose(1, 0), "z")
        pointcloud = pointcloud.transpose(1, 0)
        label = np.copy(self.label[item])
        return (pointcloud, label)
    
    def __len__(self):  # Add this method
        return len(self.pc)


# ==================
# select_target_data
# ==================
def select_target_by_conf(trgt_train_loader, model=None):
    pc_list = []
    label_list = []
    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            data = data[1].to(device)
            data = data.permute(0, 2, 1)

            logits = model(data)
            cls_conf = sfm(logits['pred'])
            mask = torch.max(cls_conf, 1)  # 2 * b
            index = 0
            for i in mask[0]:
                if i > args.threshold:
                    pc_list.append(data[index].cpu().numpy())
                    label_list.append(mask[1][index].cpu().numpy())
                index += 1
    return pc_list, label_list


# ==================
# Data loader
# ==================
src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset

# source
if src_dataset == 'modelnet':
    src_trainset = ModelNet(io, args.dataroot, 'train')
    src_testset = ModelNet(io, args.dataroot, 'test')

elif src_dataset == 'shapenet':
    src_trainset = ShapeNet(io, args.dataroot, 'train')
    src_testset = ShapeNet(io, args.dataroot, 'test')

elif src_dataset == 'scannet':
    src_trainset = ScanNet(io, args.dataroot, 'train')
    src_testset = ScanNet(io, args.dataroot, 'test')

else:
    io.cprint('unknown src dataset')

# target
if trgt_dataset == 'modelnet':
    trgt_trainset = ModelNet(io, args.dataroot, 'train')
    trgt_testset = ModelNet(io, args.dataroot, 'test')

elif trgt_dataset == 'shapenet':
    trgt_trainset = ShapeNet(io, args.dataroot, 'train')
    trgt_testset = ShapeNet(io, args.dataroot, 'test')

elif trgt_dataset == 'scannet':
    trgt_trainset = ScanNet(io, args.dataroot, 'train')
    trgt_testset = ScanNet(io, args.dataroot, 'test')
else:
    io.cprint('unknown trgt dataset')

src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size, sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size, sampler=src_valid_sampler)
src_test_loader = DataLoader(src_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size, sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size, sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)


# ==================
# Init Model
# ==================
model = DGCNN(args)
model = model.to(device)
path = args.out_path + '/{}_{}_{}_{}'.format(args.src_dataset, 
                                            args.trgt_dataset, 
                                            args.seed, args.type_) + 'pcfea.pt'
model.load_state_dict(torch.load(path))
model = model.to(device)
best_model = None

# ==================
# Optimizer
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(opt, args.epochs)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch

def entropy(*c):
    result = -1
    if len(c) > 0:
        result = 0
    for x in c:
        result += (-x) * math.log(x, 2)
    return result

src_val_acc_list = []
src_val_loss_list = []
trgt_val_acc_list = []
trgt_val_loss_list = []

def self_train(trgt_new_train_loader, src_train_loader, src_val_loader, trgt_val_loader, model=None):
    count = 0.0
    src_print_losses = {'cls': 0.0}
    trgt_print_losses = {'cls': 0.0}
    global spl_weight
    global cls_weight
    for epoch in range(args.epochs):

        model.train()

        for data1, data2 in zip(trgt_new_train_loader, src_train_loader):
            opt.zero_grad()
            batch_size = data1[1].size()[0]
            t_data, t_labels = data1[0].to(device), data1[1].to(device)
            t_logits = model(t_data)
            loss_t = spl_weight * criterion(t_logits["pred"], t_labels)
            trgt_print_losses['cls'] += loss_t.item() * batch_size
            loss_t.backward()
            s_data, s_labels = data2[1].to(device), data2[2].to(device)
            s_data = s_data.permute(0, 2, 1)
            s_logits = model(s_data)
            loss_s = cls_weight * criterion(s_logits["pred"], s_labels)
            src_print_losses['cls'] += loss_s.item() * batch_size
            loss_s.backward()
            count += batch_size
            opt.step()

        spl_weight -= 5e-3  # 0.005
        cls_weight -= 5e-3  # 0.005
        scheduler.step()

        src_print_losses = {k: v * 1.0 / count for (k, v) in src_print_losses.items()}
        io.print_progress("Source", "Trn", epoch, src_print_losses)
        trgt_print_losses = {k: v * 1.0 / count for (k, v) in trgt_print_losses.items()}
        io.print_progress("Target_new", "Trn", epoch, trgt_print_losses)

        # ===================
        # Validation
        # ===================
        src_val_acc, src_val_loss, _ = test(src_val_loader, model, "Source", "Val", epoch)
        trgt_val_acc, trgt_val_loss, _ = test(trgt_val_loader, model, "Target", "Val", epoch)
        src_val_acc_list.append(src_val_acc)
        src_val_loss_list.append(src_val_loss)
        trgt_val_acc_list.append(trgt_val_acc)
        trgt_val_loss_list.append(trgt_val_loss)
        with open('finetune_convergence.json', 'w') as f:
            json.dump((src_val_acc_list, src_val_loss_list, trgt_val_acc_list, trgt_val_loss_list), f)


# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):

    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data1 in test_loader:
            data, labels = data1[1].to(device), data1[2].to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits = model(data)
            loss = criterion(logits["pred"], labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["pred"].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat


trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_test_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))
if trgt_test_acc > 0.9:
    args.threshold = 0.95
trgt_new_best_val_acc = 0

for i in range(10):
    trgt_select_data = select_target_by_conf(trgt_train_loader, model)
    trgt_new_data = DataLoad(io, trgt_select_data)
    trgt_new_train_loader = DataLoader(trgt_new_data, num_workers=NWORKERS, batch_size=args.batch_size, drop_last=True)
    self_train(trgt_new_train_loader, src_train_loader, src_val_loader, trgt_val_loader, model)
    trgt_new_val_acc, _, _ = test(src_val_loader, model, "Source", "Val", 0)
    test(trgt_test_loader, model, "Target", "Test", 0)
    if trgt_new_val_acc > trgt_new_best_val_acc:
        trgt_new_best_val_acc = trgt_new_val_acc
        io.save_model(model, "SPST")
    args.threshold += 5e-3

path = args.out_path + '/{}_{}_{}_SPST'.format(args.src_dataset, 
                                              args.trgt_dataset, 
                                              args.seed) + args.model_type +'.pt'
model.load_state_dict(torch.load(path))
trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_test_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))