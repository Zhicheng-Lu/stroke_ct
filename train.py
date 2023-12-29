import os
import sys, getopt
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from data_reader import DataReader
from modules.segmentation import Segmentation
from datetime import datetime


def main(argv):
	data_reader = DataReader()

	# Read in command line args
	task = ''
	step = ''
	opts, args = getopt.getopt(argv,"ht:s:",["task=","step="])
	for opt, arg in opts:
		if opt == '-h':
			print ('train.py -t <taskname (segmentation/classification/severity)> -s <step (train/test)>')
			sys.exit()
		elif opt in ("-t", "--task"):
			task = arg
		elif opt in ("-s", "--step"):
			step = arg


	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")
	time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	if task == 'segmentation' and (step == 'train' or step == 'training'):
		segmentation_train(data_reader, device, time)
	if task == 'segmentation' and (step == 'test' or step == 'testing'):
		segmentation_test(data_reader, device, time)


def segmentation_train(data_reader, device, time):
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = dice_loss()
	model = Segmentation().to(device)
	print(model)

	optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)

	for epoch in range(data_reader.epochs):
		patient_range, cts, masks = data_reader.read_in_batch('segmentation', 'train')
		# masks = masks[:,:,:,None]
		# for i in range(len(cts)):
		# 	cv2.imwrite(f'temp/{i}_ct.png', cts[i]*255)
		# 	cv2.imwrite(f'temp/{i}_mask.png', masks[i]*255)
		# return
		cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
		masks = torch.from_numpy(masks)
		for iteration in range(data_reader.iterations):
			model.train()
			cts = cts.to(device=device, dtype=torch.float)
			pred = model(device, patient_range, cts)
			masks = masks.type(torch.cuda.LongTensor)
			masks.to(device)

			loss = entropy_loss_fn(pred, masks) + dice_loss_fn(pred, masks)

			# Backpropagation
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

			loss = loss.item()
			print(f"Epoch {epoch+1} iteration {iteration+1} loss: {loss:>7f}")

		torch.save(model.state_dict(), 'checkpoints/segmentation_model_{}.pt'.format(time))



def segmentation_test(data_reader, device, time):
	loss_fn = nn.CrossEntropyLoss()
	model = Segmentation()
	model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	model = model.to(device)

	patient_range, cts, masks = data_reader.read_in_batch('segmentation', 'test')
	cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
	cts = cts.to(device=device, dtype=torch.float)
	pred = model(device, patient_range, cts)
	masks = torch.from_numpy(masks)
	masks = masks.type(torch.cuda.LongTensor)
	masks.to(device)
	loss = loss_fn(pred, masks)
	loss = loss.item()
	print(f"loss: {loss:>7f}")

	output = pred.cpu().detach().numpy()
	output = np.argmax(output, axis=1)
	print(np.max(output))
	output = output[:, :, :, None]
	# output = np.repeat(output, 3, axis=3)
	output = output * 255
	for i,img in enumerate(output):
		cv2.imwrite('test/{}.jpg'.format(i), img)


class dice_loss(torch.nn.Module):
	def __init__(self):
		super(dice_loss, self).__init__()

	def forward(self, pred, masks):
		pred_softmax = nn.functional.softmax(pred, dim=1).float()
		pred_masks = pred_softmax[:,1,:,:]
		overlap = pred_masks * masks
		area_pred = torch.sum(pred_masks, dim=(1,2))
		area_masks = torch.sum(masks, dim=(1,2))
		area_overlap = torch.sum(overlap, dim=(1,2))

		# print(torch.sum(area_pred), torch.sum(area_masks), torch.sum(area_overlap))

		loss = 1 - (2 * area_overlap + 1) / (area_pred + area_masks + 1)
		loss = torch.mean(loss)
		return loss



if __name__ == "__main__":
	main(sys.argv[1:])
