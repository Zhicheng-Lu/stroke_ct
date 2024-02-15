import os
import cv2
import numpy as np
import re
import torch
from torch import nn
import configparser
from models.segmentation import Segmentation
from models.classification import Classification
from scipy.ndimage import zoom



def main():
	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	# Read in configuration
	data_reader = DataReader()

	# Read input from test_input/
	imgs = []
	img_files = os.listdir('test_input')
	img_files = sorted(img_files, key=lambda s: int(re.sub(r'\D', '', s) or 0))
	for img_file in img_files:
		img = cv2.imread(f'test_input/{img_file}')
		img = cv2.resize(img, (data_reader.height, data_reader.width))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.reshape(img, (data_reader.height, data_reader.width, 1))
		imgs.append(img / 255)

	imgs = np.array(imgs)
	imgs = np.moveaxis(imgs, 3, 1)

	# Load segmentation model
	segmentation_model = Segmentation(data_reader.f_size)
	segmentation_model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"))
	segmentation_model = segmentation_model.to(device)

	# Apply segmentation model
	segmentation_imgs = torch.from_numpy(imgs)
	segmentation_imgs = segmentation_imgs.to(device=device, dtype=torch.float)

	with torch.no_grad():
		segmentation_pred = segmentation_model(device, segmentation_imgs)
		segmentation_pred = nn.functional.softmax(segmentation_pred, dim=1)
		segmentation_pred = torch.argmax(segmentation_pred, dim=1)
		segmentation_pred = segmentation_pred.detach().cpu().numpy()
		
	# Print out
	for i,img in enumerate(segmentation_pred):
		img = img[:,:,None]
		img = img * 255
		cv2.imwrite(f'test_output/{i}.jpg', img)




	# classification
	area = np.sum(segmentation_pred)
	if area == 0:
		classification_pred = 0

	else:
		# Load model
		classification_model = Classification(data_reader)
		classification_model.load_state_dict(torch.load("checkpoints/classification_model.pt"))
		classification_model = classification_model.to(device)

		classification_imgs = zoom(imgs, (data_reader.num_slices/imgs.shape[0], 1, 1, 1))
		classification_imgs = torch.from_numpy(classification_imgs[None,:])
		classification_imgs = classification_imgs.to(device=device, dtype=torch.float)

		# Predict
		with torch.no_grad():
			classification_pred = classification_model(device, classification_imgs, data_reader)
			classification_pred = nn.functional.softmax(classification_pred, dim=0)
			classification_pred = classification_pred.detach().cpu().numpy()
			classification_pred = np.argmax(classification_pred) + 1

	categories = ['Normal', 'Ischemic Stroke', 'Hemorrhagic Stroke']




	# Severity
	if classification_pred == 0:
		severity == 'No stroke'

	else:
		if classification_pred == 1:
			centers = torch.load('checkpoints/severity_ischemic.pt', weights_only=True)
			clusters = ['Moderate to severe stroke', 'Minor stroke', 'Severe stroke', 'Moderate stroke']
		else:
			centers = torch.load('checkpoints/severity_hemorrhagic.pt', weights_only=True)
			clusters = ['Moderate stroke', 'Moderate to severe stroke', 'Minor stroke', 'Severe stroke']

		centers = centers.detach().cpu().numpy()
		shape = segmentation_pred.shape
		severity_input = zoom(segmentation_pred, (8/shape[0], 50/shape[1], 50/shape[2]))
		severity_input = severity_input.flatten()
		
		dists = []
		for center in centers:
			dist = np.sum((center - severity_input) * (center - severity_input))
			dists.append(dist)

		dists = np.array(dists)
		cluster = clusters[np.argmin(dists)]


	print(f'Category: {categories[classification_pred]}')
	print(f'Severity: {cluster}')



class DataReader():
	def __init__(self):
		super(DataReader, self).__init__()
		# Parse all arguments
		config = configparser.ConfigParser()
		config.read('config.ini')
		self.config = config

		# Image
		self.width = int(config['Image']['width'])
		self.height = int(config['Image']['height'])
		self.num_slices = int(config['Image']['num_slices'])

		# Model
		self.f_size = int(config['Model']['f_size'])




if __name__ == "__main__":
	main()