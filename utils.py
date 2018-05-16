import os
import sys
import pickle
import config as cfg
import numpy as np
import ipdb
import math
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def ACC(gt, pred):
  from sklearn.metrics import confusion_matrix as cm
  if type(gt)==list: gt = np.array(gt)
  if type(pred)==list: pred = np.array(pred)
  # ipdb.set_trace()
  acc = cm(gt, pred)
  acc_norm = acc.astype('float') / acc.sum(axis=1)[:, np.newaxis]
  aca = np.diag(acc_norm).mean()

  return acc, aca

def ACC_TEST(config, data_loader, mode = 'TEST', verbose=True):
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  PREDICTION = []
  GROUNDTRUTH = []
  total_idx=int(len(data_loader)/config.batch_size)  
  count = 0
  loss = []
  for i, (real_x, org_c, files) in enumerate(data_loader):

    if mode!='VAL' and os.path.isfile(config.pkl_data.format(mode.lower())): 
      PREDICTION, GROUNDTRUTH = pickle.load(open(config.pkl_data.format(mode.lower())))
      break
    # ipdb.set_trace()
    real_x = config.to_var(real_x, volatile=True)
    labels = org_c
  
    out_temp = config.C(real_x)
    # output = ((F.sigmoid(out_cls_temp)>=0.5)*1.).data.cpu().numpy()
    # ipdb.set_trace()
    _, output = torch.max(out_temp, 1)
    # loss.append(F.cross_entropy(out_temp, config.to_var(org_c).squeeze(1)))
    loss.append(config.LOSS(out_temp, config.to_var(org_c).squeeze(1)))
    if i==0 and verbose:
      print(mode.upper())
      print("Predicted:   "+str(output.data.cpu().numpy().tolist()))
      print("Groundtruth: "+str(org_c.cpu().numpy().flatten().tolist()))

    count += org_c.shape[0]
    if verbose:
      string_ = str(count)+' / '+str(len(data_loader)*config.batch_size)
      sys.stdout.write("\r%s" % string_)
      sys.stdout.flush()    
    # ipdb.set_trace()

    PREDICTION.extend(output.data.cpu().numpy().tolist())
    GROUNDTRUTH.extend(labels.cpu().numpy().astype(np.uint8).tolist())

  if mode!='VAL' and not os.path.isfile(config.pkl_data.format(mode.lower())): 
    pickle.dump([PREDICTION, GROUNDTRUTH], open(config.pkl_data.format(mode.lower()), 'w'))
  if verbose: 
    print("")
    print >>config.f, ""
  # ipdb.set_trace()

  PREDICTION = np.array(PREDICTION)
  GROUNDTRUTH = np.array(GROUNDTRUTH)

  acc, aca = ACC(GROUNDTRUTH, PREDICTION)

  string_ = "ACA: %0.4f"%(aca)
  if verbose: 
    print(string_)
    print >>config.f, string_ 

  return acc, aca, np.array(loss).mean()


def pdf2png(filename):
  from wand.image import Image
  from wand.color import Color
  with Image(filename="{}.pdf".format(filename), resolution=500) as img:
    with Image(width=img.width, height=img.height, background=Color("white")) as bg:
      bg.composite(img,0,0)
      bg.save(filename="{}.png".format(filename))
  os.remove('{}.pdf'.format(filename))