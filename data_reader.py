import os
import cv2
import configparser
import numpy as np
import random
import re

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

		# Data
		segmentation_path = config['Data']['segmentation']
		segmentation_folders = np.load(segmentation_path)

		# Model
		self.f_size = int(config['Model']['f_size'])

		# Training process
		test_set = int(config['Train']['test_set'])
		self.segmentation_epochs = int(config['Train']['segmentation_epochs'])
		self.classification_epochs = int(config['Train']['classification_epochs'])
		self.batch_size = int(config['Train']['batch_size'])

		# Split training and testing folders
		train_folders = [0,1,2,3,4]
		train_folders.remove(test_set)
		folders = {'train': train_folders, 'test': [test_set]}
		self.segmentation_folders = {'train': [], 'test': []}

		for mode in ['train', 'test']:
			for folder in folders[mode]:
				datasets_path = os.path.join('data', 'segmentation', f'folder_{folder}', mode)
				datasets = os.listdir(datasets_path)
				for dataset in datasets:
					dataset_path = os.path.join(datasets_path, dataset)
					patients = os.listdir(os.path.join(dataset_path, 'images'))
					for patient in patients:
						cts_path = os.path.join(dataset_path, 'images', patient)
						masks_path = os.path.join(dataset_path, 'masks', patient)
						self.segmentation_folders[mode].append((cts_path, masks_path))



	def read_in_batch(self, task, step, epoch):
		if task == 'segmentation' and (step == 'train' or step == 'training' or step == 'test' or step == 'testing'):
			return self.read_in_batch_segmentation(cts_path, masks_path)
		if task == 'classification' and (step == 'train' or step == 'training' or step == 'test' or step == 'testing'):
			return self.read_in_batch_classification()


	def read_in_batch_segmentation(self, cts_path, masks_path):
		batches_imgs = []
		batches_masks = []

		# Find all image files in the directory, and sort
		img_dir = cts_path
		img_files = os.listdir(img_dir)
		img_files = [f for f in img_files if os.path.isfile(os.path.join(img_dir, f))]
		img_files = sorted(img_files, key=lambda s: int(re.sub(r'\D', '', s) or 0))
		
		# For each image file in the directory
		for i,img_file in enumerate(img_files):
			img_file_path = os.path.join(img_dir, img_file)
			img = cv2.imread(img_file_path)
			# Resize to 512*512
			img = cv2.resize(img, (self.height, self.width))
			# Convert to greyscale
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# Add extra channel
			img = np.reshape(img, (self.height, self.width, 1))
			
			batches_imgs.append(img / 255)

			# Read masks
			mask_file_path = os.path.join(masks_path, img_file)
			if not os.path.exists(mask_file_path):
				batches_masks.append(np.zeros((self.height, self.width)))
			else:
				mask = cv2.imread(mask_file_path)
				mask = cv2.resize(mask, (self.height, self.width))
				mask = np.where(mask > 0.5, 1, 0)
				mask = np.min(mask, axis=2)
				batches_masks.append(mask)

		batches_imgs = np.array(batches_imgs)
		batches_masks = np.array(batches_masks)
		shape = batches_imgs.shape
		if shape[0] > 40:
			batches_imgs = np.resize(batches_imgs, (40, shape[1], shape[2], shape[3]))
			batches_masks = np.resize(batches_masks, (40, shape[1], shape[2]))

		return batches_imgs, batches_masks



	def read_in_batch_classification(self):
		patient_range = []
		batches_imgs = []
		batches_labels = []

		# Data
		hemorrhagic_dir = self.config['Data']['segmentation_hemorrhagic_dir']
		hemorrhagic_patients = os.listdir(os.path.join(hemorrhagic_dir, 'mask'))
		ischemic_dir = self.config['Data']['segmentation_ischemic_dir']
		ischemic_patients = os.listdir(os.path.join(ischemic_dir, 'mask'))
		dirs = {'Hemorrhagic': hemorrhagic_dir, 'Ischemic': ischemic_dir}

		# Randomly find batches for hemorrhagic and ischemic stroke
		batches = {'Hemorrhagic': random.sample(hemorrhagic_patients, int(self.batch_size / 2)), 'Ischemic': random.sample(ischemic_patients, int(self.batch_size / 2))}
		# batches = {'Hemorrhagic': ['049'], 'Ischemic': []}
		for stroke_type in ['Hemorrhagic', 'Ischemic']:
			for batch in batches[stroke_type]:
				# Find all image files in the directory, and sort
				img_dir = os.path.join(dirs[stroke_type], 'images', batch)
				img_files = os.listdir(img_dir)
				img_files = [f for f in img_files if os.path.isfile(os.path.join(dirs[stroke_type], 'images', batch, f))]
				img_files = sorted(img_files, key=lambda s: int(re.sub(r'\D', '', s) or 0))
				# Record number of slices for each patient
				if len(patient_range) == 0:
					patient_range.append(len(img_files))
				else:
					patient_range.append(patient_range[-1] + len(img_files))
				
				# For each image file in the directory
				for i,img_file in enumerate(img_files):
					img_file_path = os.path.join(dirs[stroke_type], 'images', batch, img_file)
					img = cv2.imread(img_file_path)
					# Resize to 512*512
					img = cv2.resize(img, (self.height, self.width))
					# Convert to greyscale
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					# Add extra channel
					img = np.reshape(img, (self.height, self.width, 1))
					
					batches_imgs.append(img / 255)

				# Read labels
				if stroke_type == 'Normal':
					batches_labels.append(0)
				elif stroke_type == 'Ischemic':
					batches_labels.append(1)
				else:
					batches_labels.append(2)

		return patient_range, np.array(batches_imgs), np.array(batches_labels)