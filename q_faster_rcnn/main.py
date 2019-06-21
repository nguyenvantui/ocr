import torch
import argparse
from datasets import qdataset
from tqdm import tqdm
from model.vgg_backbone import vgg_backbone
import time
import torchsummary

# hello world
parser = argparse.ArgumentParser(description='faster-rcnn')
parser.add_argument('--batch', dest='batch_size',
                    help='batch_size',
                    default=3, type=int)
parser.add_argument('--data_train', dest='data_train',
                    help='data_train',
                    default="../../", type=str)
parser.add_argument('--shape', dest='shape',
                    help='shape',
                    default=600, type=int)
parser.add_argument('--epochs', dest='epochs',
                    help='epochs',
                    default=10, type=int)

parser.add_argument('--display', dest='display',
                    help='display',
                    default=10, type=int)

parser.add_argument('--opt', dest='opt',
                    help='optimizer',
                    default="sgd", type=str)

parser.add_argument('--lr_max', dest='lr_max',
                    help='lr_max',
                    default=0.0001, type=float)

parser.add_argument('--lr_min', dest='lr_min',
                    help='lr_min',
                    default=0.00001, type=float)


args = parser.parse_args()
print(args)
# ====================== prepare data and model faster rcnn ===============================
device = "cuda"
train_data = qdataset("training", args.shape, args.data_train, 100)
train_load = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=args.batch_size,
                                         shuffle=True)

faster_rcnn = vgg_backbone(classes=10, pretrained=True)
faster_rcnn.build()
faster_rcnn = faster_rcnn.to(device)

# torchsummary.summary(faster_rcnn)
# ==========================================================================================

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def optimizer_loader():
    params = []
    lr = args.lr_max
    for key, value in dict(faster_rcnn.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

    if args.opt == "adam":
        return torch.optim.Adam(params)
    elif args.opt == "sgd":
        return torch.optim.SGD(params, momentum=0.9)

optimizer = optimizer_loader()

def train(epoch):
    print("Training epoch:",epoch)
    train_loss = 0
    now = time.time()
    for idx, (data, info, gt, num) in enumerate(train_load):
        data, info, gt, num = data.to(device), info.to(device), gt.to(device), num.to(device)
        faster_rcnn.zero_grad()

        _, _, _, rpn_loss_cls, rpn_loss_box, rcnn_loss_cls, rcnn_loss_bbox, rois_label = faster_rcnn(data, info, gt, num)
        loss = rpn_loss_cls + rpn_loss_box + rcnn_loss_cls + rcnn_loss_bbox
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % args.display == 0:
            loss_now = (train_loss / (idx + 1))
            time_now = time.time() - now
            total_data = len(train_data)
            done_data = idx * args.batch_size
            now = time.time()
            print("Data: {} / {} Loss:{:.7f}  Time:{:.2f} s".format(done_data, total_data, loss_now, time_now))

        clip_gradient(faster_rcnn, 10.0)


def test(epoch):
    # print()
    pass

def main():
    print("================================================================")
    print("Training batch size:", args.batch_size)

    for epoch in range(args.epochs):
        faster_rcnn.train()
        train(epoch)
        faster_rcnn.eval()
        test(epoch)

if __name__ == "__main__":
    print(">>> run run <<<")
    # main()
