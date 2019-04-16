import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
from pathlib import Path

class EvalDataset(Dataset):
    """
    A Pytorch Dataset Class to load images from our evaluation folder
    """
    def __init__(self):
        self.images = list(Path('../TestingPicturesOnModel').glob('*'))
    
    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        image, _, _, _ = transform(image, None, None, None, 'TEST')
        return image
    
    def __len__(self):
        return len(self.images)
    
    def collate_fn(self, batch):
        images = list()

        for b in batch:
            images.append(b[0])

        images = torch.stack(images, dim=0)
        return images

class PKLotFromDF(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, dataframe,  split):
        self.dataframe = dataframe
        self.split = split.upper()
        
    def __getitem__(self, i):
        # Read image
        image = Image.open('../project'/self.dataframe.iloc[i, 0] , mode='r')
        image = image.convert('RGB')
                              
        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = torch.FloatTensor(self.dataframe.iloc[i, 2])  # (n_objects, 4)
        labels = torch.LongTensor(self.dataframe.iloc[i, 1])  # (n_objects)
        lables = labels + 1
        
         # Apply transformations
        image, boxes, labels, _ = transform(image, boxes, labels, None, split=self.split)

        return image, boxes, labels
    
    def __len__(self):
        return len(self.dataframe.index)
    
    def compose(self, image, occupied, rotated_bboxes):
        for func in self.transform:
            image, occupied, rotated_bboxes = func(image, occupied, rotated_bboxes)
        return image, occupied, rotated_bboxes
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
