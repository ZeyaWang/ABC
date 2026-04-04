from torchvision.transforms.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, CenterCrop
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from easydl import FileListDataset
from os.path import join
from config import *


class DatasetInfo:
    def __init__(self, root, domain_list, file_list, img_prefix):
        self.path = root
        self.prefix = img_prefix
        self.domains = domain_list
        self.files = [join(root, f) for f in file_list]
        self.prefixes = [self.prefix] * len(self.domains)


domain_map = {
    'office':     ['amazon', 'dslr', 'webcam'],
    'officehome': ['Art', 'Clipart', 'Product', 'Real_World'],
    'visda':      ['train'],
}

data_roots = {
    'office':     'data/office',
    'officehome': 'data/OfficeHome',
    'visda':      'data/visda',
}

if args.dataset == 'office':
    ds_info = DatasetInfo(
        root=data_roots[args.dataset],
        domain_list=['amazon', 'dslr', 'webcam'],
        file_list=['amazon.txt', 'dslr.txt', 'webcam.txt'],
        img_prefix=data_roots[args.dataset],
    )
elif args.dataset == 'officehome':
    ds_info = DatasetInfo(
        root=data_roots[args.dataset],
        domain_list=['Art', 'Clipart', 'Product', 'Real_World'],
        file_list=['Art.txt', 'Clipart.txt', 'Product.txt', 'Real_world.txt'],
        img_prefix=data_roots[args.dataset],
    )
elif args.dataset == 'visda':
    ds_info = DatasetInfo(
        root=data_roots[args.dataset],
        domain_list=['train', 'validation'],
        file_list=['train.txt', 'validation.txt'],
        img_prefix=data_roots[args.dataset],
    )
    ds_info.prefixes = [join(ds_info.path, 'train'), join(ds_info.path, 'validation')]
else:
    raise Exception(f'dataset {args.dataset} is not supported')


source_domain_name = ds_info.domains[args.source]
target_domain_name = ds_info.domains[args.target]
src_file = ds_info.files[args.source]
tgt_file = ds_info.files[args.target]


if args.target_type == 'OPDA':
    cls_shared   = {'office': 10, 'officehome': 10, 'visda': 6}
    cls_src_priv = {'office': 10, 'officehome':  5, 'visda': 3}
    cls_total    = {'office': 31, 'officehome': 65, 'visda': 12}

elif args.target_type == 'OSDA':
    cls_shared   = {'office': 10, 'officehome': 25, 'visda': 6}
    cls_src_priv = {'office':  0, 'officehome':  0, 'visda': 0}
    cls_total    = {'office': 21, 'officehome': 65, 'visda': 12}


n_shared   = cls_shared[args.dataset]
n_src_priv = cls_src_priv[args.dataset]
n_tgt_priv = cls_total[args.dataset] - n_shared - n_src_priv

shared_cls   = [i for i in range(n_shared)]
src_priv_cls = [i + n_shared for i in range(n_src_priv)]

if args.dataset == 'office' and args.target_type == 'OSDA':
    tgt_priv_cls = [i + n_shared + n_src_priv + 10 for i in range(n_tgt_priv)]
else:
    tgt_priv_cls = [i + n_shared + n_src_priv for i in range(n_tgt_priv)]

source_classes = shared_cls + src_priv_cls
target_cls     = shared_cls + tgt_priv_cls
num_src_cls    = len(source_classes)


imagenet_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

aug_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    imagenet_norm,
])

eval_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    imagenet_norm,
])


tgt_test_ds = FileListDataset(
    list_path=tgt_file,
    path_prefix=ds_info.prefixes[args.target],
    transform=eval_transform,
    filter=(lambda x: x in target_cls),
)

target_test_dl = DataLoader(
    dataset=tgt_test_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    sampler=None,
)

target_train_ds = FileListDataset(
    list_path=tgt_file,
    path_prefix=ds_info.prefixes[args.target],
    transform=eval_transform,
    filter=(lambda x: x in target_cls),
)

target_train_ds.labels = [i for i in range(len(target_train_ds.datas))]
target_train_dl = DataLoader(
    dataset=target_train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=3,
    drop_last=True,
)
