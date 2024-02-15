import os
import cv2
import configparser
import numpy as np
import random
import re
from scipy.ndimage import zoom


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
		segmentation_train_set = eval(config['Train']['segmentation_train_set'])
		segmentation_test_set = eval(config['Train']['segmentation_test_set'])
		self.segmentation_epochs = int(config['Train']['segmentation_epochs'])
		self.classification_train_set = eval(config['Train']['classification_train_set'])
		self.classification_test_set = eval(config['Train']['classification_test_set'])
		self.classification_epochs = int(config['Train']['classification_epochs'])
		self.batch_size = int(config['Train']['batch_size'])

		# Split training and testing folders (segmentation)
		folders = {'train': segmentation_train_set, 'test': segmentation_test_set}
		self.segmentation_folders = {'train': [], 'test': []}

		for mode in ['train', 'test']:
			for folder in folders[mode]:
				datasets_path = os.path.join('data', 'segmentation', f'{folder}', mode)
				datasets = os.listdir(datasets_path)
				for dataset in datasets:
					dataset_path = os.path.join(datasets_path, dataset)
					patients = os.listdir(os.path.join(dataset_path, 'images'))
					for patient in patients:
						cts_path = os.path.join(dataset_path, 'images', patient)
						masks_path = os.path.join(dataset_path, 'masks', patient)
						self.segmentation_folders[mode].append((cts_path, masks_path, dataset))
		# Shuffle for random order
		random.shuffle(self.segmentation_folders['train'])
		random.shuffle(self.segmentation_folders['test'])

		# Split training and testing folders (classification)
		folders = {'train': self.classification_train_set, 'test': self.classification_test_set}
		self.classification_folders = {'train': [], 'test': []}

		for mode in ['train', 'test']:
			for folder in folders[mode]:
				datasets_path = os.path.join('data', 'classification', f'{folder}', mode)
				datasets = os.listdir(datasets_path)
				for dataset in datasets:
					dataset_path = os.path.join(datasets_path, dataset)
					patients = os.listdir(dataset_path)
					for patient in patients:
						cts_path = os.path.join(dataset_path, patient)
						self.classification_folders[mode].append((cts_path, patient, dataset))
		# Shuffle for random order
		random.shuffle(self.classification_folders['train'])
		random.shuffle(self.classification_folders['test'])


		# Severity prediction
		severity_paths = {'hemorrhagic': config['Train']['severity_hemorrhagic'], 'ischemic': config['Train']['severity_ischemic']}
		self.severity = {'hemorrhagic': [], 'ischemic': []}
		for severity_type, severity_path in severity_paths.items():
			datasets = os.listdir(severity_path)
			for dataset in datasets:
				for patient in os.listdir(f'{severity_path}/{dataset}/images'):
					self.severity[severity_type].append((f'{severity_path}/{dataset}/images/{patient}', f'{severity_path}/{dataset}/masks/{patient}'))



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
		# Resize to 40 if more than 40 slices
		shape = batches_imgs.shape
		if shape[0] > 40:
			batches_imgs = zoom(batches_imgs, (40/shape[0], 1, 1, 1))
			batches_masks = zoom(batches_masks, (40/shape[0], 1, 1))

		return batches_imgs, batches_masks



	# Read labels and store them in dictionary for future reference
	def prepare_labels_classification(self):
		self.labels_dict = {}

		datasets = os.listdir('data/classification/labels')
		for dataset_txt in datasets:
			dataset = dataset_txt.split('.')[0]
			self.labels_dict[dataset] = {}
			f = open(f'data/classification/labels/{dataset}.txt', 'r')
			for row in f:
				sample_name = row.strip().split('\t')[0]
				label = row.strip().split('\t')[1]
				self.labels_dict[dataset][sample_name] = eval(label)
			f.close()


	def read_in_batch_classification(self, cts_path, patient, dataset):
		cts = []

		ct_files = os.listdir(cts_path)
		ct_files = [f for f in ct_files if os.path.isfile(os.path.join(cts_path, f))]
		ct_files = sorted(ct_files, key=lambda s: int(re.sub(r'\D', '', s) or 0))

		# For each image file in the directory
		for i,ct_file in enumerate(ct_files):
			ct_file_path = os.path.join(cts_path, ct_file)
			ct = cv2.imread(ct_file_path)
			# Resize to 512*512
			ct = cv2.resize(ct, (self.height, self.width))
			# Convert to greyscale
			ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
			# Add extra channel
			ct = np.reshape(ct, (self.height, self.width, 1))
			
			cts.append(ct / 255)

		# Resize to pre-defined value
		cts = np.array(cts)
		shape = cts.shape
		if shape[0] != self.num_slices:
			cts = zoom(cts, (self.num_slices/shape[0], 1, 1, 1))
		cts = np.moveaxis(cts, 3, 1)


		# Labels
		label = self.labels_dict[dataset][patient.split('_')[0]]
		label = np.array(label)
		label = np.argmax(label)

		return cts, label



	def read_in_batch_severity(self, cts_path, masks_path):
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
		# Resize to 40 if more than 40 slices
		shape = batches_imgs.shape
		batches_imgs = zoom(batches_imgs, (self.num_slices/shape[0], 1, 1, 1))
		batches_imgs = np.moveaxis(batches_imgs, 3, 1)
		batches_masks = zoom(batches_masks, (self.num_slices/shape[0], 1, 1))

		return batches_imgs, batches_masks