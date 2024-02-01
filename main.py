import os
import sys, getopt
import torch
from datetime import datetime
from data_reader import DataReader
from segmentation import segmentation_train, segmentation_test, Diceloss
from classification import classification_train, classification_test
# from severity import severity_train


def main(argv):
	data_reader = DataReader()

	# Read in command line args
	task = ''
	step = ''
	opts, args = getopt.getopt(argv,"ht:s:",["task=","step="])
	if len(opts) == 0:
		print('Use python main.py -h for help')
		sys.exit()
	for opt, arg in opts:
		if opt == '-h':
			print ('main.py -t <taskname (segmentation/classification/severity)> -s <step (train/test)>')
			sys.exit()
		elif opt in ("-t", "--task"):
			task = arg
		elif opt in ("-s", "--step"):
			step = arg


	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")
	# torch.cuda.empty_cache()
	time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	# Call the corresponding function
	if task == 'segmentation' and (step == 'train' or step == 'training'):
		segmentation_train(data_reader, device, time)
	if task == 'segmentation' and (step == 'test' or step == 'testing'):
		segmentation_test(data_reader, device, time)
	if task == 'classification' and (step == 'train' or step == 'training'):
		classification_train(data_reader, device, time)
	if task == 'classification' and (step == 'test' or step == 'testing'):
		classification_test(data_reader, device, time)
	if task == 'severity' and (step == 'train' or step == 'training'):
		severity_train(data_reader, device, time)





if __name__ == "__main__":
	main(sys.argv[1:])
