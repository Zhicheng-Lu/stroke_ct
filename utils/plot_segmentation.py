import os
import numpy as np
import cv2
import re
from scipy.ndimage import zoom


def main():
	stroke_type = 'Ischemic'
	dataset = 'AISD'
	patient = '0538942'

	imgs = []
	masks = []

	files = os.listdir(f'../data/severity/{stroke_type}/{dataset}/images/{patient}')
	files = sorted(files, key=lambda s: int(re.sub(r'\D', '', s) or 0))
	for file in files:
		img = cv2.resize(cv2.imread(f'../data/severity/{stroke_type}/{dataset}/images/{patient}/{file}'), (80, 80))
		imgs.append(img)

		mask_file_path = f'../data/severity/{stroke_type}/{dataset}/masks/{patient}/{file}'
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

	for i,img in enumerate(imgs):
		cv2.imwrite(f'../test/severity/{i}.png', img)


if __name__ == "__main__":
	main()