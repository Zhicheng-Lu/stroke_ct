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

		# Image
		self.width = int(config['Image']['width'])
		self.height = int(config['Image']['height'])
		self.num_slices = int(config['Image']['num_slices'])

		# Data
		hemorrhagic_dir = config['Data']['hemorrhagic_dir']
		self.hemorrhagic_patients = os.listdir(os.path.join(hemorrhagic_dir, 'mask'))
		ischemic_dir = config['Data']['ischemic_dir']
		self.ischemic_patients = os.listdir(os.path.join(ischemic_dir, 'mask'))
		self.dirs = {'Hemorrhagic': hemorrhagic_dir, 'Ischemic': ischemic_dir}

		# Training process
		self.epochs = int(config['Train']['epochs'])
		self.iterations = int(config['Train']['iterations'])
		self.batch_size = int(config['Train']['batch_size'])


	def read_in_batch(self, task, step):
		if task == 'segmentation' and (step == 'train' or step == 'training' or step == 'test' or step == 'testing'):
			return self.read_in_batch_segmentation()


	def read_in_batch_segmentation(self):
		patient_range = []
		batches_imgs = []
		batches_masks = []

		# Randomly find batches for hemorrhagic and ischemic stroke
		batches = {'Hemorrhagic': random.sample(self.hemorrhagic_patients, int(self.batch_size / 2)), 'Ischemic': random.sample(self.ischemic_patients, int(self.batch_size / 2))}
		batches = {'Hemorrhagic': ['049'], 'Ischemic': []}
		for stroke_type in ['Hemorrhagic', 'Ischemic']:
			for batch in batches[stroke_type]:
				# Find all image files in the directory, and sort
				img_dir = os.path.join(self.dirs[stroke_type], 'images', batch)
				img_files = os.listdir(img_dir)
				img_files = [f for f in img_files if os.path.isfile(os.path.join(self.dirs[stroke_type], 'images', batch, f))]
				img_files = sorted(img_files, key=lambda s: int(re.sub(r'\D', '', s) or 0))
				# Record number of slices for each patient
				if len(patient_range) == 0:
					patient_range.append(len(img_files))
				else:
					patient_range.append(patient_range[-1] + len(img_files))
				
				# For each image file in the directory
				for i,img_file in enumerate(img_files):
					img_file_path = os.path.join(self.dirs[stroke_type], 'images', batch, img_file)
					img = cv2.imread(img_file_path)
					# Resize to 512*512
					img = cv2.resize(img, (self.height, self.width))
					# Convert to greyscale
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					# Add extra channel
					img = np.reshape(img, (self.height, self.width, 1))
					
					batches_imgs.append(img / 255)

					# Read masks
					basename = os.path.splitext(img_file)[0]
					extension = os.path.splitext(img_file)[1]
					if stroke_type == 'Hemorrhagic':
						mask_file_path = os.path.join(self.dirs[stroke_type], 'mask', batch, '{}_HGE_Seg{}'.format(basename, extension))
					else:
						mask_file_path = os.path.join(self.dirs[stroke_type], 'mask', batch, img_file)
					if not os.path.exists(mask_file_path):
						batches_masks.append(np.zeros((self.height, self.width)))
					else:
						mask = cv2.imread(mask_file_path)
						mask = cv2.resize(mask, (self.height, self.width))
						mask = np.where(mask > 0.5, 1, 0)
						mask = np.min(mask, axis=2)
						batches_masks.append(mask)

		return patient_range, np.array(batches_imgs), np.array(batches_masks)