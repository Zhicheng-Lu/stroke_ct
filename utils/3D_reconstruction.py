import os
import cv2
import re
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def main():
	cts_path = '../temp/samples/segmentation_samples/BHSD_015/images'
	cts = os.listdir(cts_path)
	cts = sorted(cts, key=lambda s: int(re.sub(r'\D', '', s) or 0))

	for ct in cts:
		print(ct)



if __name__ == "__main__":
	main()