import os
import numpy as np
import ipdb
import pandas as pd
import urllib, cStringIO
import Image
import matplotlib.pyplot as plt
import tqdm 
def get_classes(sheet):
	classes=sheet.keys()
	classes.pop(3)#remove url
	assert 'url' not in classes
	classes.insert(0,'neutral')	
	classes_to_idx = {k:i for i,k in enumerate(classes)}
	idx_to_classes = {i:k for i,k in enumerate(classes)}
	return 	classes_to_idx, idx_to_classes	

def get_data():
	xls = pd.ExcelFile('../data/EmotionNet2/URLsWithEmotion.xls')
	sheet = xls.parse(0)
	sheet={k.encode("utf-8"): v for k,v in sheet.iteritems()}
	classes_to_idx, idx_to_classes = get_classes(sheet)	
	return sheet, classes_to_idx, idx_to_classes

def get_label(sheet, idx):
	classes_to_idx, idx_to_classes = get_classes(sheet)	
	for classes in classes_to_idx.keys():
		if classes=='neutral': continue
		if sheet[classes][idx]==1: 
			return classes_to_idx[classes]
	return classes_to_idx['neutral']

if __name__=='__main__':
	sheet, classes_to_idx, idx_to_classes = get_data()
	folder = '../data/EmotionNet2/Images'
	url_imgs = sheet['url']
	hist_labels = '../data/EmotionNet2/hist.png'
	labels = []
	not_found = 0
	for idx in tqdm.tqdm(range(len(url_imgs))):
		label = get_label(sheet, idx)
		img_file = os.path.join(folder, '{}_{}.jpg'.format(str(label).zfill(2), str(idx).zfill(4)))

		url_img = sheet['url'][idx].encode('utf-8')
		try: 
			# ipdb.set_trace()
			if not os.path.isfile(img_file):
				Image.open(cStringIO.StringIO(urllib.urlopen(url_img).read())).save(img_file)
			labels.append(label)
		except IOError: 
			print("Url (idx excel {}) not found: {}".format(idx+2, url_img))
			not_found+=1

	print("-- # Images not found: "+str(not_found))
	plt.hist(np.histogram(labels), facecolor='green')
	plt.savefig(hist_labels)




