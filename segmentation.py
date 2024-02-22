import os
import cv2
import numpy as np
import torch
from torch import nn
from data_reader import DataReader
from torch.cuda import amp
from models.segmentation import Segmentation


def segmentation_train(data_reader, device, time):
	# Define loss and model
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()
	model = Segmentation(data_reader.f_size)
	# model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"))
	model = model.to(device)

	# Define optimier and scaler
	optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)

	losses = []

	os.mkdir(f'checkpoints/segmentation_model_{time}')

	for epoch in range(data_reader.segmentation_epochs):
		optimizer.zero_grad(set_to_none=True)
		# Train
		train_loss = 0.0
		for iteration, (cts_path, masks_path, _) in enumerate(data_reader.segmentation_folders['train']):
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

			# Gradient accumulation
			if (iteration+1) % data_reader.batch_size == 0:
				scaler.step(optimizer)
				optimizer.zero_grad(set_to_none=True)
				scaler.update()

		train_loss = train_loss / len(data_reader.segmentation_folders['train'])

		torch.save(model.state_dict(), f'checkpoints/segmentation_model_{time}/epoch_{str(epoch).zfill(3)}.pt')

		# Test
		test_loss = 0.0
		AISD = 0.0
		AISD_count = 0
		BCIHM = 0.0
		BCIHM_count = 0
		for iteration, (cts_path, masks_path, _) in enumerate(data_reader.segmentation_folders['test']):
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

				if 'AISD' in cts_path:
					AISD += loss.item()
					AISD_count += 1
				if 'BCIHM' in cts_path:
					BCIHM += loss.item()
					BCIHM_count += 1

		test_loss = test_loss / len(data_reader.segmentation_folders['test'])
		AISD_loss = AISD / AISD_count
		BCIHM_loss = BCIHM / BCIHM_count

		losses.append((epoch, train_loss, test_loss, AISD_loss, BCIHM_loss))

		print(losses)





def segmentation_test(data_reader, device, time, write_to_file=True):
	# Define loss function and model
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()
	model = Segmentation(data_reader.f_size)
	model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"))
	model = model.to(device)

	# Define metrics and initalize empty
	metrics = ['Dice', 'IOU', 'precision', 'recall']
	results = {matrix: {'overall': {'TP': 0, 'FP': 0, 'FN': 0}} for matrix in metrics}

	if write_to_file:
		os.mkdir(f'test/segmentation_{time}')
		f = open(f'test/segmentation_{time}/log.txt', 'a')

	for iteration, (cts_path, masks_path, dataset) in enumerate(data_reader.segmentation_folders['test']):
		# Add dataset to certain metrics
		if not dataset in results['Dice']:
			for matrix in results:
				results[matrix][dataset] = {'TP': 0, 'FP': 0, 'FN': 0}

		cts_np, masks = data_reader.read_in_batch_segmentation(str(cts_path), str(masks_path))

		cts = torch.from_numpy(np.moveaxis(cts_np, 3, 1))

		cts = cts.to(device=device, dtype=torch.float)

		# Infarct level evaluation metrics
		with torch.no_grad():
			pred = model(device, cts)
			pred_softmax = nn.functional.softmax(pred, dim=1).float()
			pred_masks = torch.argmax(pred_softmax, dim=1)
			pred_masks = pred_masks.detach().cpu().numpy()
			# pred_masks = pred_softmax[:,1,:,:]
			overlap = pred_masks * masks
			area_pred = np.sum(pred_masks)
			area_masks = np.sum(masks)
			TP = np.sum(overlap)
			FP = area_pred - TP
			FN = area_masks - TP


		# Write output to file
		if write_to_file:
			os.mkdir(f'test/segmentation_{time}/{iteration}_{dataset}')
			f.write(f'{iteration}_{dataset}\n')
			f.write(f'{cts_path}, {masks_path}\n')
			f.write(f'Dice: {(2*TP + 1) / (2*TP + FP + FN + 1)}\n\n')
			for i, (ct, mask, pred) in enumerate(zip(cts_np, masks, pred_masks)):
				cv2.imwrite(f'test/segmentation_{time}/{iteration}_{dataset}/{i}_ct.jpg', ct*255)
				cv2.imwrite(f'test/segmentation_{time}/{iteration}_{dataset}/{i}_gt.jpg', mask*255)
				cv2.imwrite(f'test/segmentation_{time}/{iteration}_{dataset}/{i}_predicted.jpg', pred*255)

		
		for matrix in results:
			results[matrix][dataset]['TP'] += TP
			results[matrix][dataset]['FP'] += FP
			results[matrix][dataset]['FN'] += FN
			results[matrix]['overall']['TP'] += TP
			results[matrix]['overall']['FP'] += FP
			results[matrix]['overall']['FN'] += FN

	# Final results
	for matrix in results:
		for dataset in results[matrix]:
			TP = results[matrix][dataset]['TP']
			FP = results[matrix][dataset]['FP']
			FN = results[matrix][dataset]['FN']
			if matrix == 'Dice':
				results[matrix][dataset] = (2*TP + 1) / (2*TP + FP + FN + 1)
			elif matrix == 'IOU':
				results[matrix][dataset] = (TP + 1) / (TP + FP + FN + 1)
			elif matrix == 'precision':
				results[matrix][dataset] = (TP + 1) / (TP + FP + 1)
			elif matrix == 'recall':
				results[matrix][dataset] = (TP + 1) / (TP + FN + 1)
		f.write(f'{matrix}\n{str(results[matrix])}\n')

	if write_to_file:
		f.close()


# Dice Loss Function
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