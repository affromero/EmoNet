class_names = ['neutral','happy','sad','fearful','angry','surprised','disgusted',\
        'happily surprised', 'happily disgusted','sadly fearful','sadly angry',\
        'sadly surprised','sadly disgusted','fearfully angry','fearfully surprised',\
        'fearfully disgusted','angrily surprised','angrily disgusted','disgustedly surprised',\
        'appalled','hatred','awed']

TXT_PATH='/home/afromero/datos2/EmoNet/data'

def update_folder(config, folder):
  import os
  config.log_path = os.path.join(config.log_path, folder)
  config.model_save_path = os.path.join(config.model_save_path, folder)
  config.result_save_path = os.path.join(config.result_save_path, folder)

def update_config(config):
  import os, glob, math, imageio

  folder_parameters = os.path.join(config.dataset, config.mode_data, 'fold_'+config.fold, config.finetuning)
  update_folder(config, folder_parameters)
  if config.BLUR: update_folder(config, 'BLUR')
  if config.GRAY: update_folder(config, 'GRAY')
  faces_names = 'Faces_256' if not config.mode=='aligned' else 'Faces_aligned_256'
  config.metadata_path = os.path.join(config.metadata_path, faces_names)

  if config.pretrained_model=='':
    try:
      # ipdb.set_trace()
      config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*.pth')))[-1]
      config.pretrained_model = os.path.basename(config.pretrained_model).split('.')[0]
    except:
      pass

  if config.test_model=='':
    try:
      # ipdb.set_trace()
      config.test_model = sorted(glob.glob(os.path.join(config.model_save_path, '*.pth')))[-1]
      config.test_model = os.path.basename(config.test_model).split('.')[0]
    except:
      config.test_model = ''  

  return config