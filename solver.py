import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import tqdm
from PIL import Image
import time
import datetime
import ipdb
import config as cfg
import glob
import pylab
import pickle
from utils import ACC_TEST, plot_confusion_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import filters

import warnings
warnings.filterwarnings('ignore')

class Solver(object):

  def __init__(self, data_loader, config):
    # Data loader
    self.data_loader = data_loader
    self.num_classes = data_loader.dataset.num_classes
    assert self.num_classes==22
    self.class_names = cfg.class_names

    self.image_size = config.image_size
    self.lr = config.lr
    self.beta1 = config.beta1
    self.beta2 = config.beta2

    # Training settings
    self.dataset = config.dataset
    self.num_epochs = config.num_epochs
    self.num_epochs_decay = config.num_epochs_decay
    self.batch_size = config.batch_size
    self.pretrained_model = config.pretrained_model
    self.use_tensorboard = config.use_tensorboard
    self.finetuning = config.finetuning
    self.stop_training = config.stop_training
    self.BLUR = config.BLUR
    self.GRAY = config.GRAY
    self.DISPLAY_NET = config.DISPLAY_NET

    # Test settings
    self.test_model = config.test_model
    self.metadata_path = config.metadata_path

    # Path
    self.log_path = config.log_path
    self.model_save_path = config.model_save_path
    self.result_save_path = config.result_save_path
    self.fold = config.fold
    self.mode_data = config.mode_data

    # Step size
    self.log_step = config.log_step

    #MISC
    self.GPU = config.GPU

    self.blurrandom = 0

    # Build tensorboard if use
    if config.mode!='sample':
      self.build_model()
      if self.use_tensorboard:
        self.build_tensorboard()

      # Start with trained model
      if self.pretrained_model:
        self.load_pretrained_model()

  #=======================================================================================#
  #=======================================================================================#
  def display_net(self):
  	#pip install git+https://github.com/szagoruyko/pytorchviz
  	from graphviz import Digraph
  	from torchviz import make_dot, make_dot_from_trace
	y = self.C(self.to_var(torch.randn(1,3,224,224)))
	g=make_dot(y, params=dict(self.C.named_parameters()))
	filename='network'
	g.filename=filename
	g.render()
	os.remove(filename)

	from wand.image import Image
	from wand.color import Color
	with Image(filename="{}.pdf".format(filename), resolution=500) as img:
	  with Image(width=img.width, height=img.height, background=Color("white")) as bg:
	    bg.composite(img,0,0)
	    bg.save(filename="{}.png".format(filename))
	os.remove('{}.pdf'.format(filename))
	print('Network saved at {}.png'.format(filename))

  #=======================================================================================#
  #=======================================================================================#
  def build_model(self):
    # Define a generator and a discriminator

    from models.vgg16 import Classifier
    self.C = Classifier(pretrained=self.finetuning, num_classes=self.num_classes) 

    # Optimizers
    self.optimizer = torch.optim.Adam(self.C.parameters(), self.lr, [self.beta1, self.beta2])

    # Print networks
    self.print_network(self.C, 'Classifier')

    self.LOSS = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
      self.C.cuda()

  #=======================================================================================#
  #=======================================================================================#
  def print_network(self, model, name):
    num_params = 0
    for p in model.parameters():
      num_params += p.numel()
    print(name)
    # print(model)
    print("The number of parameters: {}".format(num_params))

  #=======================================================================================#
  #=======================================================================================#
  def load_pretrained_model(self):
    model = os.path.join(
      self.model_save_path, '{}.pth'.format(self.pretrained_model))
    self.C.load_state_dict(torch.load(model))
    print('loaded trained model: {}!'.format(model))

  #=======================================================================================#
  #=======================================================================================#
  def build_tensorboard(self):
    from logger import Logger
    self.logger = Logger(self.log_path)

  #=======================================================================================#
  #=======================================================================================#
  def update_lr(self, lr):
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

  #=======================================================================================#
  #=======================================================================================#
  def reset_grad(self):
    self.optimizer.zero_grad()

  #=======================================================================================#
  #=======================================================================================#
  def to_var(self, x, volatile=False):
    if torch.cuda.is_available():
      x = x.cuda()
    return Variable(x, volatile=volatile)

  #=======================================================================================#
  #=======================================================================================#
  def threshold(self, x):
    x = x.clone()
    x = (x >= 0.5).float()
    return x

  #=======================================================================================#
  #=======================================================================================#
  def denorm(self, x):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    out = x + mean
    return out.clamp_(0, 1)

  #=======================================================================================#
  #=======================================================================================#
  def blurRANDOM(self, img):
    self.blurrandom +=1

    np.random.seed(self.blurrandom) 
    gray = np.random.randint(0,2,img.size(0))
    np.random.seed(self.blurrandom)
    sigma = np.random.randint(2,9,img.size(0))
    np.random.seed(self.blurrandom)
    window = np.random.randint(7,29,img.size(0))

    trunc = (((window-1)/2.)-0.5)/sigma
    # ipdb.set_trace()
    conv_img = torch.zeros_like(img.clone())
    for i in range(img.size(0)):    
      # ipdb.set_trace()
      if gray[i] and self.GRAY:
        conv_img[i] = torch.from_numpy(filters.gaussian_filter(img[i], sigma=sigma[i], truncate=trunc[i]))
      else:
        for j in range(img.size(1)):
          conv_img[i,j] = torch.from_numpy(filters.gaussian_filter(img[i,j], sigma=sigma[i], truncate=trunc[i]))

    return conv_img
    
  #=======================================================================================#
  #=======================================================================================#
  def plot_cm(self, CM, aca, E, i):
    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(CM, classes=self.class_names,
    #                       title='Confusion matrix, without normalization.\nACA: %0.3f'%(aca))
    # pylab.savefig(os.path.join(self.result_save_path, '{}_{}.png'.format(E, i+1)))

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(CM, classes=self.class_names, normalize=True,
                          title='CM. ACA: %0.3f'%(aca))
    pylab.savefig(os.path.join(self.result_save_path, '{}_{}_norm.png'.format(E, i+1)))   

  #=======================================================================================#
  #=======================================================================================#
  def train(self):
    # Set dataloader

    # The number of iterations per epoch
    iters_per_epoch = len(self.data_loader)
    data_loader = self.data_loader

    # lr cache for decaying
    lr = self.lr
    
    # Start with trained model if exists
    if self.pretrained_model:
      start = int(self.pretrained_model.split('_')[0])
      # Decay learning rate
      for i in range(start):
        if (i+1) > (self.num_epochs - self.num_epochs_decay):
          # g_lr -= (self.g_lr / float(self.num_epochs_decay))
          lr -= (self.lr / float(self.num_epochs_decay))
          self.update_lr(lr)
          print ('Decay learning rate to: {}.'.format(lr))      
    else:
      start = 0

    last_model_step = len(self.data_loader)

    print("Log path: "+self.log_path)

    Log = "[EmoNets] bs:{}, fold:{}, GPU:{}, !{}, from:{}".format(self.batch_size, self.fold, self.GPU, self.mode_data, self.finetuning) 
    loss_cum = {}
    loss_cum['LOSS'] = []
    flag_init=True   

    loss_val_prev = 90
    aca_val_prev = 0
    non_decreasing = 0

    # Start training
    start_time = time.time()

    for e in range(start, self.num_epochs):
      E = str(e+1).zfill(2)
      self.C.train()

      if flag_init:
        CM, aca_val, loss_val = self.val(init=True)  
        log = '[ACA_VAL: %0.3f LOSS_VAL: %0.3f]'%(aca_val, loss_val)
        print(log)
        flag_init = False
        if self.pretrained_model:
          aca_val_prev=aca_val
        self.plot_cm(CM, aca_val, E, 0) 

      for i, (rgb_img, rgb_label, rgb_files) in tqdm.tqdm(enumerate(self.data_loader), \
          total=len(self.data_loader), desc='Epoch: %d/%d | %s'%(e,self.num_epochs, Log)):
        # ipdb.set_trace()
        if self.BLUR: rgb_img = self.blurRANDOM(rgb_img)

        rgb_img = self.to_var(rgb_img)
        rgb_label = self.to_var(rgb_label)

        out = self.C(rgb_img)

        loss_cls = self.LOSS(out, rgb_label.squeeze(1))   

        # # Backward + Optimize
        self.reset_grad()
        loss_cls.backward()
        self.optimizer.step()


        # Logging
        loss = {}
        loss['LOSS'] = loss_cls.data[0]
        loss_cum['LOSS'].append(loss_cls.data[0])    
        # Print out log info
        if (i+1) % self.log_step == 0 or (i+1)==last_model_step:
          if self.use_tensorboard:
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)


      #F1 val
      CM, aca_val, loss_val = self.val()

      if self.use_tensorboard:
        self.logger.scalar_summary('ACC_val: ', aca_val, e * iters_per_epoch + i + 1) 
        self.logger.scalar_summary('LOSS_val: ', loss_val, e * iters_per_epoch + i + 1)     

        for tag, value in loss_cum.items():
          self.logger.scalar_summary(tag, np.array(value).mean(), e * iters_per_epoch + i + 1)   
               
      #Stats per epoch
      elapsed = time.time() - start_time
      elapsed = str(datetime.timedelta(seconds=elapsed))

      log = 'Elapsed: %s | [ACC_VAL: %0.3f LOSS_VAL: %0.3f] | Train'%(elapsed, aca_val, loss_val)
      for tag, value in loss_cum.items():
        log += ", {}: {:.4f}".format(tag, np.array(value).mean())   

      print(log)

      # if loss_val<loss_val_prev:
      if aca_val>aca_val_prev:        
        torch.save(self.C.state_dict(), os.path.join(self.model_save_path, '{}_{}.pth'.format(E, i+1)))
        print("! Saving model")
        # Compute confusion matrix
        np.set_printoptions(precision=2)
        self.plot_cm(CM, aca_val, E, i+1)       

        # loss_val_prev = loss_val
        aca_val_prev = aca_val
        non_decreasing = 0

      else:
        non_decreasing+=1
        if non_decreasing == self.stop_training:
          print("During {} epochs ACC VAL was not increasing.".format(self.stop_training))
          return

      # Decay learning rate
      if (e+1) > (self.num_epochs - self.num_epochs_decay):
        lr -= (self.lr / float(self.num_epochs_decay))
        self.update_lr(lr)
        print ('Decay learning rate to: {}.'.format(lr))

  #=======================================================================================#
  #=======================================================================================#
  def val(self, init=False, load=False, plot=False):
    # Load trained parameters
    if init:
      from data_loader import get_loader
      # ipdb.set_trace()
      self.data_loader_val = get_loader(self.metadata_path, self.image_size,
                   self.image_size, self.batch_size, self.fold, 'EmotionNet', 'val')

      txt_path = os.path.join(self.model_save_path, '0_init_val.txt')
    
    if load:
      self.data_loader_val = self.data_loader
      last_file = sorted(glob.glob(os.path.join(self.model_save_path,  '*.pth')))[-1]
      last_name = os.path.basename(last_file).split('.')[0]
      txt_path = os.path.join(self.model_save_path, '{}_{}_val.txt'.format(last_name,'{}'))
      try:
        output_txt  = sorted(glob.glob(txt_path.format('*')))[-1]
        number_file = len(glob.glob(output_txt))
      except:
        number_file = 0
      txt_path = txt_path.format(str(number_file).zfill(2)) 

      D_path = os.path.join(self.model_save_path, '{}.pth'.format(last_name))
      self.C.load_state_dict(torch.load(D_path))

    self.C.eval()

    if load: self.f=open(txt_path, 'a')
    acc,aca,loss = ACC_TEST(self, self.data_loader_val, mode='VAL', verbose=load)
    if load: self.f.close()

    if plot: 
      np.set_printoptions(precision=2)
      self.plot_cm(acc, aca, int(last_name.split('_')[0]), int(last_name.split('_')[1]))  

    return acc, aca, loss

  #=======================================================================================#
  #=======================================================================================#
  def sample(self):
    """Get a dataset sample."""
    import math
    for i, (rgb_img, rgb_label, rgb_files) in enumerate(self.data_loader):
        # ipdb.set_trace()
        if self.BLUR: rgb_img = self.blurRANDOM(rgb_img)
        img_file = 'show/%s.jpg'%(str(i).zfill(4))
        save_image(self.denorm(rgb_img), img_file, nrow=8)
        if i==25: break



