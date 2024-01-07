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

		# Model
		self.f_size = int(config['Model']['f_size'])

		# Training process
		self.segmentation_epochs = int(config['Train']['segmentation_epochs'])
		self.segmentation_iterations = int(config['Train']['segmentation_iterations'])
		self.classification_epochs = int(config['Train']['classification_epochs'])
		self.classification_iterations = int(config['Train']['classification_iterations'])
		self.batch_size = int(config['Train']['batch_size'])


	def read_in_batch(self, task, step, epoch):
		if task == 'segmentation' and (step == 'train' or step == 'training' or step == 'test' or step == 'testing'):
			return self.read_in_batch_segmentation(epoch)
		if task == 'classification' and (step == 'train' or step == 'training' or step == 'test' or step == 'testing'):
			return self.read_in_batch_classification()


	def read_in_batch_segmentation(self, epoch):
		patient_range = []
		batches_imgs = []
		batches_masks = []

		# Data
		hemorrhagic_dir = self.config['Data']['segmentation_hemorrhagic_dir']
		# hemorrhagic_patients = os.listdir(os.path.join(hemorrhagic_dir, 'mask'))
		ischemic_dir = self.config['Data']['segmentation_ischemic_dir']
		# ischemic_patients = os.listdir(os.path.join(ischemic_dir, 'mask'))
		dirs = {'Hemorrhagic': hemorrhagic_dir, 'Ischemic': ischemic_dir}

		# Randomly find batches for hemorrhagic and ischemic stroke
		batches = {}
		for stroke_type in ['Hemorrhagic', 'Ischemic']:
			datasets = os.listdir(dirs[stroke_type])
			dataset = datasets[random.randint(0, len(datasets)-1)]
			patients = os.listdir(os.path.join(dirs[stroke_type], dataset, 'images'))
			patients = random.sample(patients, int(self.batch_size / 2))
			batches[stroke_type] = (dataset, patients)


		for stroke_type in ['Hemorrhagic', 'Ischemic']:
			for batch in batches[stroke_type][1]:
				# Find all image files in the directory, and sort
				datasets_dir = dirs[stroke_type]
				dataset = batches[stroke_type][0]
				img_dir = os.path.join(datasets_dir, dataset, 'images', batch)
				img_files = os.listdir(img_dir)
				img_files = [f for f in img_files if os.path.isfile(os.path.join(img_dir, f))]
				img_files = sorted(img_files, key=lambda s: int(re.sub(r'\D', '', s) or 0))
				# Record number of slices for each patient
				if len(patient_range) == 0:
					patient_range.append((0, len(img_files)))
				else:
					patient_range.append((patient_range[-1][1], patient_range[-1][1] + len(img_files)))
				
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
					mask_file_path = os.path.join(datasets_dir, dataset, 'masks', batch, img_file)
					if not os.path.exists(mask_file_path):
						batches_masks.append(np.zeros((self.height, self.width)))
					else:
						mask = cv2.imread(mask_file_path)
						mask = cv2.resize(mask, (self.height, self.width))
						mask = np.where(mask > 0.5, 1, 0)
						mask = np.min(mask, axis=2)
						batches_masks.append(mask)

		return patient_range, np.array(batches_imgs), np.array(batches_masks)



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