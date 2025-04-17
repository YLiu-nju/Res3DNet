import SimpleITK
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import xlrd
import random
import monai.transforms as MT

def repeat_alpha(dataset, alpha):
    result = dataset
    while alpha > 0:
        result = result + dataset
        alpha -= 1
    return result


def create_folds(root_dir, excel_name, fold_num, alpha=2):
    fold_file_name = os.path.dirname(root_dir) + '/{0:d}-fold-partition.txt'.format(fold_num)
    folds = {}
    if os.path.exists(fold_file_name):
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in folds:
                    folds[fold_id] = []
                folds[fold_id].append([params[1], params[2]])
    else:
        dataset_1 = []
        dataset_0 = []
        test_set_tcga = []
        test_set_all = []

        wb = xlrd.open_workbook(excel_name)
        sheet = wb.sheet_by_name('train-internal')
        for i in range(sheet.nrows):
            if i == 0:
                continue
            if sheet.cell_value(i, 2) == 1.0:
                dataset_1.append([sheet.cell_value(i, 3), int(sheet.cell_value(i, 2))])
            else:
                dataset_0.append([sheet.cell_value(i, 3), int(sheet.cell_value(i, 2))])

        sheet = wb.sheet_by_name('TCGA')
        for i in range(sheet.nrows):
            if i == 0:
                continue
            test_set_tcga.append([sheet.cell_value(i, 3), int(sheet.cell_value(i, 2))])

        sheet = wb.sheet_by_name('all')
        for i in range(sheet.nrows):
            if i == 0:
                continue
            test_set_all.append([sheet.cell_value(i, 3), int(sheet.cell_value(i, 2))])

        if len(dataset_1) < len(dataset_0):
            dataset_1 = repeat_alpha(dataset_1, alpha)
        else:
            dataset_0 = repeat_alpha(dataset_0, alpha)

        case_num = len(dataset_1)
        case_num0 = len(dataset_0)
        random.shuffle(dataset_1)
        random.shuffle(dataset_0)

        # training
        folds[0] = dataset_1[:int(case_num * 0.6)]
        folds[0] += dataset_0[:int(case_num0 * 0.6)]
        random.shuffle(folds[0])

        # val
        folds[1] = dataset_1[int(case_num * 0.6): int(case_num * 0.8)]
        folds[1] += dataset_0[int(case_num0 * 0.6): int(case_num0 * 0.8)]
        random.shuffle(folds[1])

        folds[2] = dataset_1[int(case_num * 0.8):]
        folds[2] += dataset_0[int(case_num0 * 0.8):]
        random.shuffle(folds[2])

        folds[3] = test_set_tcga
        random.shuffle(folds[3])

        folds[4] = test_set_all
        random.shuffle(folds[4])

    folds_size = [len(x) for x in folds.values()]
    return folds, folds_size


class Aug():
    def __init__(self):
        self.train_transforms = MT.Compose(
            [
                MT.AddChanneld(keys=['t1', 't1c', 't2', 'flair']),
                MT.Orientationd(keys=['t1', 't1c', 't2', 'flair'], axcodes="RAS"),
                MT.Spacingd(
                    keys=['t1', 't1c', 't2', 'flair'],
                    pixdim=(2.0, 1.5, 1.5),
                ),
                MT.RandFlipd(
                    keys=['t1', 't1c', 't2', 'flair'],
                    spatial_axis=[0],
                    prob=0.50,
                ),
                MT.RandFlipd(
                    keys=['t1', 't1c', 't2', 'flair'],
                    spatial_axis=[1],
                    prob=0.50,
                ),
                MT.RandFlipd(
                    keys=['t1', 't1c', 't2', 'flair'],
                    spatial_axis=[2],
                    prob=0.50,
                ),
                MT.RandRotate90d(
                    keys=['t1', 't1c', 't2', 'flair'],
                    prob=0.50,
                    max_k=3,
                ),
                MT.RandShiftIntensityd(
                    keys=['t1', 't1c', 't2', 'flair'],
                    offsets=0.10,
                    prob=0.50,
                ),
                MT.RandHistogramShiftD(keys=['t1', 't1c', 't2', 'flair'], prob=1, num_control_points=30,
                                       allow_missing_keys=True),
                MT.NormalizeIntensityd(keys=['t1', 't1c', 't2', 'flair']),
                MT.AddChanneld(keys=['t1', 't1c', 't2', 'flair']),
                MT.ToTensord(keys=['t1', 't1c', 't2', 'flair']),
            ]
        )
        self.rand_contrast = MT.RandAdjustContrast(prob=0.5, gamma=(0.5, 4.5))
        self.rand_affine = MT.RandAffined(
            keys=['t1', 't1c', 't2', 'flair'],
            prob=1.0,
            spatial_size=(267, 267),
            translate_range=(40, 40),
            rotate_range=(np.pi / 36, np.pi / 36),
            scale_range=(0.15, 0.15),
            padding_mode="border",
        )


    def forward(self, x):
        x = self.rand_affine(x)
        x['t1'] = self.rand_contrast(x['t1'])
        x['t1c'] = self.rand_contrast(x['t1c'])
        x['t2'] = self.rand_contrast(x['t2'])
        x['flair'] = self.rand_contrast(x['flair'])
        x = self.train_transforms(x)
        return x

class Test_transform():
    def __init__(self):
        self.test_transforms = MT.Compose(
            [
                MT.AddChanneld(keys=['t1', 't1c', 't2', 'flair']),
                MT.NormalizeIntensityd(keys=['t1', 't1c', 't2', 'flair']),
                MT.AddChanneld(keys=['t1', 't1c', 't2', 'flair']),
                MT.ToTensord(keys=['t1', 't1c', 't2', 'flair']),
            ]
        )

    def forward(self, x):
        x = self.test_transforms(x)
        return x

class IDHDataset(Dataset):
    def __init__(self, root_dir, ids, data_shape, dataAug=True):
        super(IDHDataset, self).__init__()
        self.root_dir = root_dir
        self.ids = ids
        self.data_shape = data_shape
        self.aug = Aug()
        self.test_trans = Test_transform()
        self.dataAug = dataAug

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        img_pth = id[0]
        label = id[1]

        t1c = SimpleITK.ReadImage(img_pth + 'T1CE.nii.gz')
        t1c = SimpleITK.GetArrayFromImage(t1c)
        t1 = SimpleITK.ReadImage(img_pth + 'T1.nii.gz')
        t1 = SimpleITK.GetArrayFromImage(t1)
        t2 = SimpleITK.ReadImage(img_pth + 'T2.nii.gz')
        t2 = SimpleITK.GetArrayFromImage(t2)
        flair = SimpleITK.ReadImage(img_pth + 'FLAIR.nii.gz')
        flair = SimpleITK.GetArrayFromImage(flair)

        if self.dataAug:
            sample = {'t1': t1, 't1c': t1c, 't2': t2, 'flair': flair}
            sample = self.aug.forward(sample)
            t1, t1c, t2, flair = sample['t1'], sample['t1c'], sample['t2'], sample['flair']
        else:
            sample = {'t1': t1, 't1c': t1c, 't2': t2, 'flair': flair}
            sample = self.test_trans.forward(sample)
            t1, t1c, t2, flair = sample['t1'], sample['t1c'], sample['t2'], sample['flair']

        t1 = F.interpolate(t1, size=self.data_shape, mode='trilinear', align_corners=False)
        t1c = F.interpolate(t1c, size=self.data_shape, mode='trilinear', align_corners=False)
        t2 = F.interpolate(t2, size=self.data_shape, mode='trilinear', align_corners=False)
        flair = F.interpolate(flair, size=self.data_shape, mode='trilinear', align_corners=False)

        data = torch.cat([t1, t1c, t2, flair], dim=1)[0]
        return {'data': data, 'label': int(label)}
