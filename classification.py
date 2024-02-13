import os
import cv2
import numpy as np
import torch
from torch import nn
from data_reader import DataReader
from torch.cuda import amp
from models.classification import Classification
from models.segmentation import Segmentation
import random


def classification_train(data_reader, device, time):
	# Prepare labels
	data_reader.prepare_labels_classification()

	# New classification model
	classification_model = Classification(data_reader)
	classification_model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	classification_model = classification_model.to(device)

	# Freeze layers
	for i,(name,param) in enumerate(classification_model.named_parameters()):
		if i < 12:
			param.requires_grad = False

	# Define optimier, scaler and loss
	optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)
	entropy_loss_fn = nn.CrossEntropyLoss()

	losses = []

	os.mkdir(f'checkpoints/classification_model_{time}')

	for epoch in range(data_reader.classification_epochs):
		# Train
		for iteration, (cts_path, patient, dataset) in enumerate(data_reader.classification_folders['train']):
			if iteration % data_reader.batch_size == 0:
				batch_cts = []
				batch_labels = []

			# remove normal class during training
			cts, label = data_reader.read_in_batch_classification(cts_path, patient, dataset)
			if label == 0:
				continue
			label -= 1

			batch_cts.append(cts)
			batch_labels.append(label)

			if (iteration+1) % data_reader.batch_size != 0:
				continue

			optimizer.zero_grad(set_to_none=True)
			
			batch_cts = np.array(batch_cts)
			batch_labels = np.array(batch_labels)
			
			batch_cts = torch.from_numpy(batch_cts).to(device=device, dtype=torch.float)
			batch_labels = torch.from_numpy(batch_labels)
			batch_labels = batch_labels.type(torch.cuda.LongTensor)
			batch_labels.to(device)

			# Train the model
			with torch.cuda.amp.autocast():
				pred = classification_model(device, batch_cts, data_reader)
				loss = entropy_loss_fn(pred, batch_labels)

			# Backpropagation
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			print(f'Epoch {epoch+1} iteration {int((iteration+1)/data_reader.batch_size)} loss: {loss.item()}')

		
		torch.save(classification_model.state_dict(), f'checkpoints/classification_model_{time}/epoch_{str(epoch).zfill(3)}.pt')


		# Test on train set
		train_losses = []
		for iteration, (cts_path, patient, dataset) in enumerate(data_reader.classification_folders['train']):
			cts, label = data_reader.read_in_batch_classification(cts_path, patient, dataset)
			if label == 0:
				continue
			label -= 1

			cts = torch.from_numpy(cts[None,:]).to(device=device, dtype=torch.float)
			label = torch.from_numpy(np.array([label]))
			label = label.type(torch.cuda.LongTensor)
			label.to(device)

			with torch.no_grad():
				pred = classification_model(device, cts, data_reader)
				loss = entropy_loss_fn(pred, label)
				train_losses.append(loss.item())
		train_loss = sum(train_losses) / len(train_losses)


		# Test on validation set
		test_losses = {'overall': []}
		for iteration, (cts_path, patient, dataset) in enumerate(data_reader.classification_folders['test']):
			# Initialize losses
			if not dataset in test_losses:
				test_losses[dataset] = []

			cts, label = data_reader.read_in_batch_classification(cts_path, patient, dataset)
			if label == 0:
				continue
			label -= 1

			cts = torch.from_numpy(cts[None,:]).to(device=device, dtype=torch.float)
			label = torch.from_numpy(np.array([label]))
			label = label.type(torch.cuda.LongTensor)
			label.to(device)

			with torch.no_grad():
				pred = classification_model(device, cts, data_reader)
				loss = entropy_loss_fn(pred, label)
				test_losses['overall'].append(loss.item())
				test_losses[dataset].append(loss.item())

		epoch_info = [epoch, train_loss]
		for dataset in test_losses:
			print(dataset)
			epoch_info.append(sum(test_losses[dataset]) / len(test_losses[dataset]))

		losses.append(tuple(epoch_info))

		print(losses)



def classification_test(data_reader, device, time):
	# Prepare labels
	data_reader.prepare_labels_classification()

	# Segmentation model
	segmentation_model = Segmentation(data_reader.f_size)
	segmentation_model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"))
	segmentation_model = segmentation_model.to(device)

	# New classification model
	classification_model = Classification(data_reader)
	classification_model.load_state_dict(torch.load("checkpoints/classification_model.pt"))
	classification_model = classification_model.to(device)

	datasets_results = {'overall': []}

	for iteration, (cts_path, patient, dataset) in enumerate(data_reader.classification_folders['test']):
		cts, label = data_reader.read_in_batch_classification(cts_path, patient, dataset)
		
		# Fed into segmentation model, to see the area of segmented lesion
		seg_cts = torch.from_numpy(cts).to(device=device, dtype=torch.float)
		with torch.no_grad():
			seg_pred = segmentation_model(device, seg_cts)
			seg_pred = torch.argmax(seg_pred, dim=1)
			area = torch.sum(seg_pred)

		if area < 5:
			pred = [1.0, 0.0, 0.0]

		else:
			# Fed into classification model
			cts = torch.from_numpy(cts[None,:]).to(device=device, dtype=torch.float)

			with torch.no_grad():
				pred = classification_model(device, cts, data_reader)
				pred = nn.functional.softmax(pred, dim=1).float()
				pred = pred.cpu().detach().numpy()[0]
				pred = np.insert(pred, 0, 0.0)

		print(label, area, pred)

		# print(label, pred)

		if not dataset in datasets_results:
			datasets_results[dataset] = []

		# if dataset == 'CQ500':
		# 	print(patient, label, pred, np.argmax(pred))

		datasets_results['overall'].append([pred, label])
		datasets_results[dataset].append([pred, label])

	for dataset, results in datasets_results.items():
		print(dataset)

		get_stats(results)



def get_stats(results):
	confusion_matrix = {}

	for stroke_type in range(1, 3):
		for num_classes in range(2, 4):
			get_confusion_matrix(results, stroke_type, num_classes)



# Confusion matrix
def get_confusion_matrix(results, stroke_type, num_classes):
	stroke_types = ['Normal', 'Ischemic', 'Hemorrhagic']

	# True positive, true negative, false positive, false negative
	TP, TN, FP, FN = 0, 0, 0, 0

	for pred, label in results:
		if num_classes == 2:
			pred_temp = np.array([-1 if i == 3 - stroke_type else pred[i] for i in range(len(pred))])
		else:
			pred_temp = pred

		# Predicted class
		pred_class = np.argmax(pred_temp)

		if label == stroke_type and pred_class == label:
			TP += 1
		elif label == stroke_type and pred_class != label:
			FP += 1
		elif label != stroke_type and pred_class != stroke_type:
			TN += 1
		else:
			FN += 1

	dice, recall, specificity = 'N/A', 'N/A', 'N/A'

	# Dice, recall and specificity
	if 2*TP + FP + FN != 0:
		dice = 2*TP / (2*TP + FP + FN)
	if TP + FN != 0:
		recall = TP / (TP+FN)
	if TN + FP != 0:
		specificity = TN / (TN + FP)
	

	# AUC
	if num_classes == 2:
		results = [[pred, label] for pred, label in results if label == 0 or label == stroke_type]

	n_pos = len([1 for pred, label in results if label == stroke_type])
	n_neg = len([1 for pred, label in results if label != stroke_type])

	results = sorted(results, key=lambda x: x[0][stroke_type], reverse=True)

	count_pos = 0
	auc = 0
	for pred, label in results:
		if n_pos == 0 or n_neg == 0:
			auc = 1
			break

		if label == stroke_type:
			count_pos += 1

		else:
			auc += count_pos / (n_pos * n_neg)

	print(f'{stroke_types[stroke_type]}_{num_classes}_AUC: {auc}')
	print(f'{stroke_types[stroke_type]}_{num_classes}_dice: {dice}')
	print(f'{stroke_types[stroke_type]}_{num_classes}_recall: {recall}')
	print(f'{stroke_types[stroke_type]}_{num_classes}_specificity: {specificity}')