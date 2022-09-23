# @Author:殷梦晗
# @Time:2022/7/27 9:13
# 将随机数据增强应用于未标记的图像 K 次，每个增强后的图像都通过分类器进行输入。
# 然后，通过调整分布的值来“锐化”这些 K 预测的平均值。
from PIL import Image
# import libraries
import math
from typing import Tuple
import wandb

# accimage安装问题（是否在win的环境下无法安装）
# import scientific libraries
import numpy as np

# import pytorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time
import torch.nn.functional as F
import pandas as pd
# set seed to reproduce results
np.random.seed(42)
# @title
# initialize wandb for logging
# !wandb login 202040aaac395bbf5a4a47d433a5335b74b7fb0e

# path of data
# 两个文件夹 covid normal（将test文件的路径放过来就可以）
# data_path = r'.\datasets\datasets'
data_path = r"D:\my_code_2\my_code\mh-metacovid-siamese-neural-network\metacovid-siamese-neural-network-main\scripts\dataset\pretrain_2c_0727\test"

CLASSES = ["COVID", "NORMAL"]
# transforms
train_transform = transforms.Compose(
    [
        transforms.Resize(100),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# datasets

ds_train = datasets.ImageFolder(str(data_path), transform=train_transform,
                                target_transform=lambda t: torch.tensor(t).long())
ds_test = datasets.ImageFolder(str(data_path), transform=test_transform,
                               target_transform=lambda t: torch.tensor(t).long())
# print(ds_train.targets)
# print(ds_train.classes)
# # print(ds_train.imgs)
# print(ds_train.class_to_idx)
# preparing data for problem 2
# Code has been adapted from GH repo - https://github.com/YU1ut/MixMatch-pytorch

# As per the problem:
# 类别数量
classes_len = 2
# 每个类别带标签的数量
n_labeled_per_class = 20


# 带标签的数量：2*20=40


def train_val_split(labels, n_labeled_per_class, classes_len):
    """
    数据划分
    有标签数据前20个，无标签数据从第21个到倒数20，验证集数据倒数20个
    :returns 标签数据、无标签数据、验证集数据列表
    """
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(classes_len):
        idxs = np.where(labels == i)[0]
        # 将所有IDs打乱
        np.random.shuffle(idxs)
        # 每类带标签的取前20个
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        # 无标签的取21-->倒数第21
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-20])
        # 验证集取倒数20个
        val_idxs.extend(idxs[-20:])
    # 再次把上述三种数据打乱
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


# 得到标签数据、无标签数据、验证集
train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(ds_train.targets, n_labeled_per_class, classes_len)


def pil_loader(path: str):
    """
    将图像进行通道转换：’RGBA‘-->’RGB‘
    :param path: 图像路径
    :return: 转换为’RGB‘的图像
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str):
    """
    快速加载图像
    :param path:图像路径
    :return: 图像
    """
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


import pandas as pd


def replace_labels(samples_list, agg_label_xlsx):
    """
    替换标签
    :param samples_list:
    :param agg_label_xlsx:
    :return:
    """
    df_train_20 = pd.DataFrame(samples_list, columns=["image", "label"])
    df_true = pd.read_excel(agg_label_xlsx)
    # """
    images_filenames = []
    for i in df_train_20["image"]:
        i = i.split("\\")[-2:]
        a = "\\".join(i)
        images_filenames.append(a)
    df_train_20["filenames"] = images_filenames
    df = pd.merge(df_train_20, df_true, on="filenames", how='left')
    # 后面真实操作的时候将true_label改成教师聚合label
    # 使用教师聚合label和加噪后的label
    df = df[["image_x", "true_label"]]
    samples = [list(x) for x in df.values]
    targets = df["true_label"].tolist()
    return samples, targets


class ChestXRay_labeled(datasets.ImageFolder):

    def __init__(self, root, indexs=None,
                 transform=None, target_transform=None,
                 loader=default_loader
                 , is_valid_file=None):
        super(ChestXRay_labeled, self).__init__(root,
                                                transform=transform, target_transform=target_transform,
                                                loader=loader,
                                                is_valid_file=is_valid_file)

        if indexs is not None:
            # self.samples = np.array(self.imgs)[indexs].tolist()
            samples = np.array(self.imgs)[indexs].tolist()
            # self.targets = np.array(self.targets)[indexs]
            agg_label_xlsx = "d.xlsx"
            self.samples, self.targets = replace_labels(samples, agg_label_xlsx)

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = super(ChestXRay_labeled, self).__getitem__(index)
    #     return img, target


# Train Dataset  img, target
train_labeled_dataset = ChestXRay_labeled(str(data_path), train_labeled_idxs, transform=train_transform,
                                          target_transform=lambda t: torch.tensor((float(t))).long())
# print(train_labeled_dataset)


# train_labeled_dataset.samples, train_labeled_dataset.targets = replace_labels(train_labeled_dataset.samples, "d.xlsx")

# print(train_labeled_dataset.samples)

# Validation and Test Datasets
val_dataset = ChestXRay_labeled(str(data_path), val_idxs, transform=test_transform,
                                target_transform=lambda t: torch.tensor((float(t))).long())
test_dataset = ChestXRay_labeled(str(data_path), transform=test_transform,
                                 target_transform=lambda t: torch.tensor((float(t))).long())
assert len(train_labeled_idxs) == 40
print(len(train_labeled_idxs))
print(len(train_unlabeled_idxs))
print(len(val_idxs))
print(len(test_dataset))


class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class ChestXRay_unlabeled(datasets.ImageFolder):

    def __init__(self, root, indexs=None,
                 transform=None, target_transform=None,
                 loader=default_loader
                 , is_valid_file=None):
        super(ChestXRay_unlabeled, self).__init__(root,
                                                  transform=transform, target_transform=target_transform,
                                                  loader=loader,
                                                  is_valid_file=is_valid_file)
        # 将标签全部更改为-1
        if indexs is not None:
            # self.samples = np.array(self.imgs)[indexs].tolist()
            samples = np.array(self.imgs)[indexs].tolist()
            # self.targets = np.array(self.targets)[indexs]
            agg_label_xlsx = "d.xlsx"
            self.samples, self.targets = replace_labels(samples, agg_label_xlsx)
        # print(self.samples)
        self.targets = np.array([-1 for i in range(len(self.targets))])
        # print(self.targets)


train_unlabeled_dataset = ChestXRay_unlabeled(str(data_path), train_unlabeled_idxs,
                                              transform=TransformTwice(train_transform))

# 设置超参数
batch_size = 16
lr = 5e-5
epochs = 2  # 30
log_freq = 10
ema_decay = 0.999
train_iteration = 2  # No of iterations per epoch  512
lambda_u = 75
T = 0.5

num_classes = len(CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')

# Train Dataloaders
# 带标签的训练数据
labeled_trainloader = DataLoader(train_labeled_dataset, batch_size=len(train_labeled_dataset), shuffle=True,
                                 pin_memory=True,
                                 drop_last=True)
# print(labeled_trainloader)

# 不带标签的训练数据
unlabeled_trainloader = DataLoader(train_unlabeled_dataset, batch_size=len(train_unlabeled_dataset), shuffle=True,
                                   pin_memory=True,
                                   drop_last=True)
# Validation dataloader
# 验证数据
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True, drop_last=True)

# Test loader on entire set of 900 images
# 测试数据
test_loader = DataLoader(test_dataset, 100000, shuffle=False, pin_memory=True, drop_last=True)

# sanity check labeled train dataloader
# 检查带标签训练数据
# torch.Size([16, 3, 224, 224])
# torch.Size([16])
# torch.int64
import pandas as pd

for batch in labeled_trainloader:
    # 根据img在excel表格中找到教师的预测标签
    img, target = batch
    # print(img)
    img = img.tolist()
    target = target.tolist()
    df = pd.DataFrame({"img": img, "target": target})
    df.to_excel("a.xlsx")
    # print(target)
    # print(img.shape)
    # print(target.shape)
    # print(target.dtype)
    break

# sanity check unlabeled train dataloader
# 检查无标签训练数据
# torch.Size([16, 3, 224, 224])
# torch.Size([16, 3, 224, 224])
for batch in unlabeled_trainloader:
    # 为什么是两张图像？
    (img1, img2), _ = batch
    print(img1.shape)
    print(img2.shape)
    break

# Class labels
# {'covid': 0, 'normal': 1}
print(train_unlabeled_dataset.class_to_idx)


# @title
# 模型的基础块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


# 模型网络块
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,
                                      activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,
                                activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


# 宽ResNet网络
class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# 创建模型
def create_model(ema=False):
    # Move the model to CUDA, if available
    #     global model
    if not torch.cuda.is_available():
        # model = WideResNet(num_classes=3)
        # 加载ResNet模型
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        # 增加线性层
        model.fc = nn.Linear(2048, num_classes)
        model.to("cpu")

    if ema == True:
        for param in model.parameters():
            # 在x->y->z传播中，如果我们对y进行detach_()，就把x->y->z切成两部分：x和y->z，x就无法接受到后面传过来的梯度
            param.detach_()
    return model


# model，ResNet
model = create_model()
# 对模型参数的传播进行处理
ema_model = create_model(ema=True)


# @title
# 线性上升
def linear_rampup(current, rampup_length=epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


# 半损失
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, lambda_u * linear_rampup(epoch)


# 权重指数移动平均值
class WeightEMA(object):
    """
    自定义权重
    """

    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                # 自定义权重衰减
                param.mul_(1 - self.wd)


# 交错偏移
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


# 交错
def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


# @title


# Macro F1 score PyTorch

class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score


# Define optimizer, loss, macro and micro F1 scores
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max = 5)
# 交叉熵
criterion = torch.nn.CrossEntropyLoss()
# 计算损失
train_criterion = SemiLoss()
# 优化器
ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay)

macro_f1_score = F1Score(average='macro')
micro_f1_score = F1Score(average='micro')

from progress.bar import Bar


# 平均
class AverageMeter(object):
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


def train(labeled_trainloader, unlabeled_trainloader, model, epoch, train_iteration=1024, T=0.5, alpha=0.75):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    # 模型训练
    model.train()
    for batch_idx in range(train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        # 测量数据加载时间
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        # 将标签转变为热编码
        targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x.view(-1, 1).long(), 1)
        #         inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        #         inputs_u = inputs_u.cuda()
        #         inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            # 计算未标记样本的猜测标签
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        # 拼接矩阵
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
        # 从β分布中提取样本
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        # 随机打乱后获得的数字序列
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        # 在批次之间交错标记和未标记的样本以获得正确的 batchnorm 计算
        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = train_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                                    epoch + batch_idx / train_iteration)

        loss = Lx + w * Lu
        # print('The loss is ', loss)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print(
            '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                batch=batch_idx + 1,
                size=train_iteration,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                w=ws.avg,
            ))

    return (losses.avg, losses_x.avg, losses_u.avg,)


def valid(val_loader, model, epoch, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    macro_f1_scores = AverageMeter()
    micro_f1_scores = AverageMeter()
    accuracies = AverageMeter()

    model.eval()
    bar = Bar(f'{mode}', max=len(val_loader))
    end = time.time()
    print(f'\n************{mode}*************')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            #       inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)

            # compute loss
            loss = criterion(outputs, targets)

            ## compute metrics
            # accuracy
            probs = torch.softmax(outputs, dim=1)
            # 得到预测标签
            predicted_labels = torch.argmax(probs, dim=1)

            correct = predicted_labels == targets
            accuracy = correct.sum() / float(targets.size(0))

            # macro F1 score
            macro_f1 = macro_f1_score(predicted_labels.flatten(), targets.flatten())

            # micro F1 score
            micro_f1 = micro_f1_score(predicted_labels.flatten(), targets.flatten())

            # update the loss and metrics
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(accuracy, inputs.size(0))
            macro_f1_scores.update(macro_f1.item(), inputs.size(0))
            micro_f1_scores.update(micro_f1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            # print the logs
            print(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Macro_F1_score: {macro_f1_score:.4f} | Micro_F1_score: {micro_f1_score:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    accuracy=accuracies.avg,
                    macro_f1_score=macro_f1_scores.avg,
                    micro_f1_score=micro_f1_scores.avg
                ))

    return losses.avg, accuracies.avg, macro_f1_scores.avg, micro_f1_scores.avg


def train_model(model, labeled_trainloader, unlabeled_trainloader, epochs, log_freq):
    print('********Training has started***************')

    wandb.watch(model, log='all')

    step = 0
    for epoch in range(1, epochs + 1):
        print('\n Epoch: [%d | %d]' % (epoch, epochs))
        # 假如总共有66条数据:20条作为训练数据,46条作为测试数据.
        # 训练:20
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, epoch=epoch,
                                                       train_iteration=train_iteration)
        # 预测46,可以得到预测标签,
        _, train_accuracy, train_macro_f1, train_micro_f1 = valid(labeled_trainloader, ema_model, epoch,
                                                                  mode='Train_stats')
        val_loss, val_accuracy, val_macro_f1, val_micro_f1 = valid(val_loader, ema_model, epoch,
                                                                   mode='Validation Stats')
        test_loss, test_accuracy, test_macro_f1, test_micro_f1 = valid(test_loader, ema_model, epoch,
                                                                       mode='Test Stats ')

        # 将带有标签的和猜测标签的数据放在一起计算准确率,为整个PATE的准确率.在此之前可计算学生机的准确率

        step = train_iteration * (epoch + 1)

        wandb.log({
            'epoch': epoch,

            # Train metrics
            'train_loss': train_loss,
            'train_loss_x': train_loss_x,
            'train_loss_u': train_loss_u,
            'train_accuracy': train_accuracy,
            'train_macro_f1': train_macro_f1,
            'train_micro_f1': train_micro_f1,

            # Validation metrics
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_macro_f1': val_macro_f1,
            'val_micro_f1': val_micro_f1,

            # Test metrics
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_macro_f1': test_macro_f1,
            'test_micro_f1': test_micro_f1,

        })

    print('**************Training has Finished**********************')

    # saving the model
    torch.save(model.state_dict(), 'model.h5')


#   wandb.save('model.h5')
# % % wandb


def main():
    # wandb initialize a new run
    wandb.init(project='Expand-ai-problem-2')
    wandb.watch_called = False

    config = wandb.config
    config.batch_size = batch_size
    config.epochs = epochs
    config.lr = lr
    config.seed = 42
    config.classes = len(CLASSES)
    config.device = device
    config.ema_decay = ema_decay
    config.train_iteration = train_iteration
    config.lambda_u = lambda_u
    config.T = T

    # set seed
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    train_model(model, labeled_trainloader, unlabeled_trainloader, epochs, log_freq)


if __name__ == '__main__':
    main()
    wandb.finish()
# '''
