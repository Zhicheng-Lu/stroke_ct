import os
import cv2
import numpy as np
import torch
from torch import nn
from data_reader import DataReader
from torch.cuda import amp
from models.segmentation_pretrained import SegmentationPreTrained
from models.classification import Classification


def classification_train(data_reader, device, time):
	# Prepare labels
	data_reader.prepare_labels_classification()

	# Pre-trained segmentation model
	segmentation_pretrained = SegmentationPreTrained(data_reader.f_size)
	segmentation_pretrained.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	segmentation_pretrained = segmentation_pretrained.to(device)

	# New classification model
	classification_model = Classification(data_reader)
	# classification_model.load_state_dict(torch.load("checkpoints/classification_model.pt"))
	classification_model = classification_model.to(device)
	optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)
	entropy_loss_fn = CrossEntropyLoss()

	losses = []

	os.mkdir(f'checkpoints/classification_model_{time}')

	for epoch in range(data_reader.classification_epochs):
		optimizer.zero_grad(set_to_none=True)
		# Train
		for iteration, (cts_path, patient, dataset) in enumerate(data_reader.classification_folders['train']):
			if iteration % data_reader.batch_size == 0:
				batch_cts = []
				batch_labels = []

			cts, label = data_reader.read_in_batch_classification(cts_path, patient, dataset)
			cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
			cts = cts.to(device=device, dtype=torch.float)
			# From pre-trained segmentation model, resize
			with torch.no_grad():
				features = segmentation_pretrained(device, cts)
				features = features.cpu().detach().numpy()
				features = np.resize(features, (data_reader.num_slices,data_reader.f_size*16,int(data_reader.height/16),int(data_reader.width/16)))
			batch_cts.append(features)
			batch_labels.append(label)

			if (iteration+1) % data_reader.batch_size != 0:
				continue
			
			batch_cts = np.array(batch_cts)
			batch_labels = np.array(batch_labels)
			
			batch_cts = torch.from_numpy(batch_cts).to(device=device, dtype=torch.float)
			batch_labels = torch.from_numpy(batch_labels)
			batch_labels = batch_labels.type(torch.cuda.LongTensor)
			batch_labels.to(device)

			# Train the model
			classification_model.train()
			with torch.cuda.amp.autocast():
				pred = classification_model(device, batch_cts)
				loss = entropy_loss_fn(pred, batch_labels)

			print(f'Epoch {epoch+1} iteration {int((iteration+1)/data_reader.batch_size)} loss: {loss.item()}')

			# Backpropagation
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			optimizer.zero_grad(set_to_none=True)
			scaler.update()

		
		torch.save(classification_model.state_dict(), f'checkpoints/classification_model_{time}/epoch_{str(epoch).zfill(3)}.pt')


		# Test on train set
		train_losses = []
		for iteration, (cts_path, patient, dataset) in enumerate(data_reader.classification_folders['train']):
			batch_cts = []
			batch_labels = []

			cts, label = data_reader.read_in_batch_classification(cts_path, patient, dataset)
			cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
			cts = cts.to(device=device, dtype=torch.float)
			# From pre-trained segmentation model, resize
			with torch.no_grad():
				features = segmentation_pretrained(device, cts)
				features = features.cpu().detach().numpy()
				features = np.resize(features, (data_reader.num_slices,data_reader.f_size*16,int(data_reader.height/16),int(data_reader.width/16)))
			batch_cts.append(features)
			batch_labels.append(label)

			# To numpy, then to torch
			batch_cts = np.array(batch_cts)
			batch_labels = np.array(batch_labels)

			batch_cts = torch.from_numpy(batch_cts).to(device=device, dtype=torch.float)
			batch_labels = torch.from_numpy(batch_labels)
			batch_labels = batch_labels.type(torch.cuda.LongTensor)
			batch_labels.to(device)

			with torch.no_grad():
				pred = classification_model(device, batch_cts)
				loss = entropy_loss_fn(pred, batch_labels)
				train_losses.append(loss.item())
		train_loss = sum(train_losses) / len(train_losses)


		# Test on test set
		test_losses = {'overall': []}
		for iteration, (cts_path, patient, dataset) in enumerate(data_reader.classification_folders['test']):
			# Initialize losses
			if not dataset in test_losses:
				test_losses[dataset] = []

			batch_cts = []
			batch_labels = []

			cts, label = data_reader.read_in_batch_classification(cts_path, patient, dataset)
			cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
			cts = cts.to(device=device, dtype=torch.float)
			# From pre-trained segmentation model, resize
			with torch.no_grad():
				features = segmentation_pretrained(device, cts)
				features = features.cpu().detach().numpy()
				features = np.resize(features, (data_reader.num_slices,data_reader.f_size*16,int(data_reader.height/16),int(data_reader.width/16)))
			batch_cts.append(features)
			batch_labels.append(label)

			# To numpy, then to torch
			batch_cts = np.array(batch_cts)
			batch_labels = np.array(batch_labels)

			batch_cts = torch.from_numpy(batch_cts).to(device=device, dtype=torch.float)
			batch_labels = torch.from_numpy(batch_labels)
			batch_labels = batch_labels.type(torch.cuda.LongTensor)
			batch_labels.to(device)

			with torch.no_grad():
				pred = classification_model(device, batch_cts)
				loss = entropy_loss_fn(pred, batch_labels)
				test_losses['overall'].append(loss.item())
				test_losses[dataset].append(loss.item())

		epoch_info = [epoch, train_loss]
		for dataset in test_losses:
			print(dataset)
			epoch_info.append(sum(test_losses[dataset]) / len(test_losses[dataset]))

		losses.append(tuple(epoch_info))

		print(losses)



def classification_test(data_reader, device, time):
	entropy_loss_fn = nn.CrossEntropyLoss()

	# Pre-trained segmentation model
	segmentation_pretrained = SegmentationPreTrained(data_reader.f_size)
	segmentation_pretrained.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	segmentation_pretrained = segmentation_pretrained.to(device)

	classification_model = Classification(data_reader)
	classification_model.load_state_dict(torch.load("checkpoints/classification_model.pt"), strict=False)
	classification_model = classification_model.to(device)

	patient_range, cts, labels = data_reader.read_in_batch('classification', 'train')
	cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
	cts = cts.to(device=device, dtype=torch.float)
	# From pre-trained segmentation model, resize
	features = segmentation_pretrained(device, patient_range, cts)
	features = features.cpu().detach().numpy()
	patient_range = [(0,patient_range[i]) if i==0 else (patient_range[i-1], patient_range[i]) for i in range(len(patient_range))]
	features = np.concatenate([np.expand_dims(np.resize(features[patient_range[i][0]:patient_range[i][1]], (data_reader.num_slices,data_reader.f_size*16,int(data_reader.height/16),int(data_reader.width/16))), axis=0) for i in range(len(patient_range))], axis=0)
	features = torch.from_numpy(features).to(device=device, dtype=torch.float)
	pred = classification_model(device, patient_range, features)

	# Labels
	labels_torch = torch.from_numpy(labels)
	labels_torch = labels_torch.type(torch.cuda.LongTensor)
	labels_torch.to(device)

	# Loss
	loss = entropy_loss_fn(pred, labels_torch)
	print(f"Loss: {loss.item():>7f}")

	# Output
	categories = ['Normal', 'Ischemic', 'Hemorrhagic']
	output = pred.cpu().detach().numpy()
	output = np.argmax(output, axis=1)
	print(f'Predicted: {[categories[i] for i in output]}')
	print(f'Labels: {[categories[i] for i in labels]}')



# Loss
class CrossEntropyLoss(torch.nn.Module):
	def __init__(self):
		super(CrossEntropyLoss, self).__init__()

	def forward(self, pred, labels):
		pred_softmax = nn.functional.softmax(pred, dim=1).float()
		# print(pred, pred_softmax, labels)
		p_metrics = pred_softmax * labels
		p_metrics = torch.sum(p_metrics, dim=1)
		p_metrics = torch.mean(p_metrics)
		return 1 - p_metrics