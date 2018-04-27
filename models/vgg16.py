from __future__ import division
import ipdb
import inspect
import os
import time
import math
import glob
import numpy as np
from six.moves import xrange
import pickle
import sys
import config as cfg
import torch.nn as nn
import torch.legacy.nn as nn_legacy
from torch.autograd import Variable
import math
import torch
# torch.backends.cudnn.enabled=False

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vgg_pytorch import vgg16 as model_vgg16

class Classifier(nn.Module):

  def __init__(self, pretrained='/npy/weights', num_classes=22):

    super(Classifier, self).__init__()
    self.pretrained=pretrained
    self.num_classes = num_classes


    self._initialize_weights()

  def _initialize_weights(self):

    if self.pretrained=='Imagenet':
      mode='Imagenet'
      self.model = model_vgg16(pretrained=True)
      modules = self.model.modules()
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1000:
          w1 = m.weight.data[1:self.num_classes+1].view(self.num_classes ,-1)
          b1 = torch.FloatTensor(np.array((m.bias.data[:self.num_classes ])).reshape(-1))
      mod = list(self.model.classifier)
      mod.pop()
      mod.append(torch.nn.Linear(4096,self.num_classes))
      new_classifier = torch.nn.Sequential(*mod)
      self.model.classifier = new_classifier
      flag=False
      modules = self.model.modules()
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==22:
          m.weight.data = w1
          m.bias.data = b1
          flag=True
      assert flag

    else:
      mode='RANDOM'
      self.model = model_vgg16(pretrained=False, num_classes=self.num_classes)  

    print("[OK] Weights initialized from %s"%(mode))
    

  def forward(self, image, OF=None):
    x = self.model(image)
    return x      