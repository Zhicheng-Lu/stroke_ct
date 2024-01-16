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
	model = Segmentation(data_reader.f_size)
	model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"))
	model = model.to(device)

	optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)

	losses = []

	os.mkdir(f'checkpoints/segmentation_model_{time}')

	for epoch in range(data_reader.segmentation_epochs):
		optimizer.zero_grad(set_to_none=True)
		# Train
		train_loss = 0.0
		for iteration, (cts_path, masks_path) in enumerate(data_reader.segmentation_folders['train']):
			print(cts_path, masks_path)
			cts, masks = data_reader.read_in_batch_segmentation(cts_path, masks_path)

			cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
			masks = torch.from_numpy(masks)

			model.train()
			cts = cts.to(device=device, dtype=torch.float)
			
			masks = masks.type(torch.cuda.LongTensor)
			masks.to(device)

			with torch.cuda.amp.autocast():
				pred = model(device, cts)
				loss = entropy_loss_fn(pred, masks) + dice_loss_fn(pred, masks)
				train_loss += loss.item()

			print(f"Epoch {epoch+1} iteration {int((iteration+2)/2)} loss: {loss}")

			# Backpropagation
			scaler.scale(loss).backward()

			if (iteration+1) % data_reader.batch_size == 0:
				scaler.step(optimizer)
				optimizer.zero_grad(set_to_none=True)
				scaler.update()

		train_loss = train_loss / len(data_reader.segmentation_folders['train'])

		torch.save(model.state_dict(), f'checkpoints/segmentation_model_{time}/epoch_{str(epoch).zfill(3)}.pt')

		# Test
		test_loss = 0.0
		for iteration, (cts_path, masks_path) in enumerate(data_reader.segmentation_folders['test']):
			cts, masks = data_reader.read_in_batch_segmentation(str(cts_path), str(masks_path))

			cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
			masks = torch.from_numpy(masks)

			cts = cts.to(device=device, dtype=torch.float)
			masks = masks.type(torch.cuda.LongTensor)
			masks.to(device)

			with torch.no_grad():
				pred = model(device, cts)
				loss = entropy_loss_fn(pred, masks) + dice_loss_fn(pred, masks)
				test_loss += loss.item()

		test_loss = test_loss / len(data_reader.segmentation_folders['test'])

		losses.append((epoch, train_loss, test_loss))

		print(losses)





def segmentation_test(data_reader, device, time):
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()

	model = Segmentation(data_reader.f_size)
	model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	model = model.to(device)

	cts_path, masks_path = data_reader.segmentation_test_folder[3]
	print(cts_path, masks_path)
	cts, masks = data_reader.read_in_batch_segmentation(str(cts_path), str(masks_path))
	# Print out input
	# masks = masks[:,:,:,None]
	for i in range(len(cts)):
		cv2.imwrite(f'test/{i}_ct.png', cts[i]*255)
		cv2.imwrite(f'test/{i}_mask.png', masks[i]*255)

	# Make prediction
	cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
	cts = cts.to(device=device, dtype=torch.float)
	masks = torch.from_numpy(masks)
	masks = masks.type(torch.cuda.LongTensor)
	masks.to(device)
	with torch.no_grad():
		pred = model(device, cts)
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