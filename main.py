#!/usr/local/bin/ipython
import os
import argparse
from data_loader import get_loader
from torch.backends import cudnn
import torch
import glob
import math
import ipdb
import imageio
import numpy as np
import config as cfg

def str2bool(v):
  return v.lower() in ('true')

def main(config):
  # For fast training
  cudnn.benchmark = True

  # Create directories if not exist
  if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)
  if not os.path.exists(config.model_save_path):
    os.makedirs(config.model_save_path)
  if not os.path.exists(config.result_save_path):
    os.makedirs(config.result_save_path)    

  img_size = config.image_size

  data_loader = get_loader(config.metadata_path, img_size,
                   img_size, config.batch_size, config.fold, 'EmotionNet', config.mode,\
                   num_workers=config.num_workers)

  # Solver
  from solver import Solver    
  solver = Solver(data_loader, config)

  if config.mode == 'train':
    solver.train()
    solver.val()
  elif config.mode == 'test':
    solver.val(load=True, plot=True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--image_size', type=int, default=224)
  parser.add_argument('--lr', type=float, default=0.0001)

  # Training settings
  parser.add_argument('--dataset', type=str, default='EmotionNet', choices=['EmotionNet'])
  parser.add_argument('--num_epochs', type=int, default=99)
  parser.add_argument('--num_epochs_decay', type=int, default=100)
  parser.add_argument('--stop_training', type=int, default=5) #How many epochs after loss_val is not decreasing
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--beta1', type=float, default=0.5)
  parser.add_argument('--beta2', type=float, default=0.999)
  parser.add_argument('--num_workers', type=int, default=4) 
  parser.add_argument('--BLUR', action='store_true', default=False) 
  parser.add_argument('--GRAY', action='store_true', default=False) 

  # Test settings
  parser.add_argument('--test_model', type=str, default='')

  # Misc
  parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
  parser.add_argument('--use_tensorboard', action='store_true', default=False)
  parser.add_argument('--GPU', type=str, default='3')

  # Path
  parser.add_argument('--metadata_path', type=str, default='/home/afromero/datos/Databases/EmotionNet')
  parser.add_argument('--log_path', type=str, default='./snapshot/logs')
  parser.add_argument('--model_save_path', type=str, default='./snapshot/models') 
  parser.add_argument('--result_save_path', type=str, default='./snapshot/results') 
  parser.add_argument('--fold', type=str, default='0', choices=['0', '1', '2', 'all'])
  parser.add_argument('--mode_data', type=str, default='normal', choices=['normal', 'aligned'])  
   
  parser.add_argument('--finetuning', type=str, default='Imagenet', choices=['Imagenet', 'RANDOM'])   
  parser.add_argument('--pretrained_model', type=str, default='')    

  # Step size
  parser.add_argument('--log_step', type=int, default=2000)
  parser.add_argument('--model_save_step', type=int, default=20000)

  config = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

  config = cfg.update_config(config)


  print(config)
  main(config)