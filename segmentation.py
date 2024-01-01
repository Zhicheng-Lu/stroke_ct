import os
import cv2
import numpy as np
import torch
from torch import nn
from data_reader import DataReader
from torch.cuda import amp
from models.segmentation import Segmentation


def segmentation_train(data_reader, device, time):
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()
	model = Segmentation(data_reader.f_size).to(device)

	optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)

	for epoch in range(data_reader.segmentation_epochs):
		patient_range, cts, masks = data_reader.read_in_batch('segmentation', 'train')
		cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
		masks = torch.from_numpy(masks)
		for iteration in range(data_reader.segmentation_iterations):
			model.train()
			optimizer.zero_grad(set_to_none=True)
			cts = cts.to(device=device, dtype=torch.float)
			
			masks = masks.type(torch.cuda.LongTensor)
			masks.to(device)

			with torch.cuda.amp.autocast():
				pred = model(device, patient_range, cts)
				loss = entropy_loss_fn(pred, masks) + dice_loss_fn(pred, masks)

			# Backpropagation
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			loss = loss.item()
			print(f"Epoch {epoch+1} iteration {iteration+1} loss: {loss:>7f}")

		torch.save(model.state_dict(), 'checkpoints/segmentation_model_{}.pt'.format(time))



def segmentation_test(data_reader, device, time):
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()

	model = Segmentation(data_reader.f_size)
	model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	model = model.to(device)

	patient_range, cts, masks = data_reader.read_in_batch('segmentation', 'test')
	# Print out input
	# masks = masks[:,:,:,None]
	for i in range(len(cts)):
		cv2.imwrite(f'test/{i}_ct.png', cts[i]*255)
		cv2.imwrite(f'test/{i}_mask.png', masks[i]*255)

	# Make prediction
	cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
	cts = cts.to(device=device, dtype=torch.float)
	pred = model(device, patient_range, cts)
	masks = torch.from_numpy(masks)
	masks = masks.type(torch.cuda.LongTensor)
	masks.to(device)
	entropy_loss = entropy_loss_fn(pred, masks)
	dice_loss = dice_loss_fn(pred, masks)
	loss = entropy_loss + dice_loss
	print(f"Entropy loss: {entropy_loss.item():>7f}")
	print(f"Dice loss: {dice_loss.item():>7f}")
	print(f"Total loss: {loss.item():>7f}")

	output = pred.cpu().detach().numpy()
	output = np.argmax(output, axis=1)
	output = output[:, :, :, None]
	output = output * 255
	for i,img in enumerate(output):
		cv2.imwrite('test/{}_output.jpg'.format(i), img)


class Diceloss(torch.nn.Module):
	def __init__(self):
		super(Diceloss, self).__init__()

	def forward(self, pred, masks):
		pred_softmax = nn.functional.softmax(pred, dim=1).float()
		pred_masks = pred_softmax[:,1,:,:]
		overlap = pred_masks * masks
		area_pred = torch.sum(pred_masks)
		area_masks = torch.sum(masks)
		area_overlap = torch.sum(overlap)

		loss = 1 - (2 * area_overlap + 1) / (area_pred + area_masks + 1)
		return loss