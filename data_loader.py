import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb
import numpy as np
import imageio
import scipy.misc
import glob

class EmotionNet(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, fold, shuffling = False):
    # ipdb.set_trace()
    self.transform = transform
    self.shuffling = shuffling
    self.image_size = image_size
    self.mode=mode
    self.images = sorted(glob.glob(metadata_path+'/*.jpg'))
    num_subjects = len(list(set([os.path.basename(line).split('.')[0].split('_')[1] for line in self.images])))
    self.num_classes = len(list(set([os.path.basename(line).split('_')[0] for line in self.images])))
    # ipdb.set_trace()
    if fold=='all':
      self.lines = self.images
    else:
      fold = int(fold)
      subjects_fold = range(int((num_subjects*fold)/3), int(num_subjects*(fold+1)/3))#Three fold crossvalidation
      if self.mode=='train':
        self.lines = [line for line in self.images if int(os.path.basename(line).split('.')[0].split('_')[1]) not in subjects_fold]
      else:
        self.lines= [line for line in self.images if int(os.path.basename(line).split('.')[0].split('_')[1]) in subjects_fold]

    self.subjects = list(set([os.path.basename(line).split('.')[0].split('_')[1] for line in self.lines]))
    # ipdb.set_trace()
    print ('Start preprocessing dataset: %s'%(self.mode))
    random.seed(1234)
    self.preprocess()
    print("# Subjects %d out of %d | Len data %d out of %d"%(len(self.subjects), num_subjects, len(self.lines), len(self.images)))
    print ('Finished preprocessing dataset: %s'%(self.mode))
    
    self.num_data = len(self.filenames)

  def preprocess(self):
    self.filenames = []
    self.labels = []
    if self.shuffling: random.shuffle(self.lines)   # random shuffling
    for line in self.lines:
      label = int(os.path.basename(line).split('_')[0])-1

      self.filenames.append(line)
      self.labels.append([label])

  def __getitem__(self, index):
    image = Image.open(self.filenames[index])
    label = self.labels[index]
    # ipdb.set_trace()
    return self.transform(image), torch.LongTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data

def get_loader(metadata_path, crop_size, image_size, batch_size, \
        fold, dataset='EmotionNet', mode='train', num_workers=4):
  """Build and return data loader."""

  #IMageNet Normalization
  # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0])

  if mode == 'train' or mode == 'sample':
    transform = transforms.Compose([
      transforms.CenterCrop((crop_size)),
      # transforms.Resize((image_size, image_size), interpolation=Image.ANTIALIAS),
      transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
      # transforms.Normalize(mean, std),
      ])  

  else:
    transform = transforms.Compose([
      # transforms.CenterCrop(crop_size),
      transforms.Resize((image_size, image_size), interpolation=Image.ANTIALIAS),
      # transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
      transforms.ToTensor(),
      normalize,
      # transforms.Normalize(mean, std),
      ])

  dataset = EmotionNet(image_size, metadata_path, transform, mode, fold, \
              shuffling=mode=='train' or mode=='sample')

  data_loader = DataLoader(dataset=dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers)
  return data_loader
