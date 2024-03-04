import os
import numpy as np
import cv2
import re
from scipy.ndimage import zoom


def main():
	path = '../test/3D_modelling/samples/severity_samples'
	folder = 'ICH_1_minor'

	imgs = []
	masks = []

	files = os.listdir(f'{path}/{folder}/images')
	files = sorted(files, key=lambda s: int(re.sub(r'\D', '', s) or 0))
	for file in files:
		img = cv2.resize(cv2.imread(f'{path}/{folder}/images/{file}'), (80, 80))
		imgs.append(img)

		mask_file_path = f'{path}/{folder}/masks/{file}'
		# if not os.path.exists(mask_file_path):
		# 	mask = np.zeros((img.shape[0], img.shape[1], 3))
		# else:
		# 	mask = cv2.resize(cv2.imread(mask_file_path), (256, 256))
		mask = cv2.resize(cv2.imread(mask_file_path), (80, 80))
		masks.append(mask)

	# imgs = zoom(imgs, (8/len(imgs), 1, 1, 1))
	# masks = zoom(masks, (8/len(imgs), 1, 1, 1))
	imgs, masks = np.array(imgs), np.array(masks)

	for sli in range(imgs.shape[0]):
		for row in range(imgs.shape[1]):
			for col in range(imgs.shape[2]):
				if np.sum(masks[sli, row, col]) > 0:
					imgs[sli, row, col] = [0, 0 ,255]

	os.mkdir(f'../test/severity/{folder}')

	for i,img in enumerate(imgs):
		cv2.imwrite(f'../test/severity/{folder}/{i}.png', img)

	# folder = '../test/3D_modelling/samples/segmentation_samples/BHSD_015/masks'
	# for file in os.listdir(folder):
	# 	img = cv2.imread(f'{folder}/{file}')
	# 	for row in range(img.shape[0]):
	# 		for col in range(img.shape[1]):
	# 			if np.sum(img[row, col]) > 0:
	# 				img[row, col] = [255, 255, 255]

	# 	img = cv2.resize(img, (80, 80))

	# 	cv2.imwrite(f'{file}', img)


if __name__ == "__main__":
	main()