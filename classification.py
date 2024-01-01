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
	# Pre-trained segmentation model
	segmentation_pretrained = SegmentationPreTrained(data_reader.f_size)
	segmentation_pretrained.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	segmentation_pretrained = segmentation_pretrained.to(device)

	# New classification model
	classification_model = Classification(data_reader).to(device)

	for epoch in range(data_reader.classification_epochs):
		patient_range, cts, masks = data_reader.read_in_batch('classification', 'train')
		cts = torch.from_numpy(np.moveaxis(cts, 3, 1))
		cts = cts.to(device=device, dtype=torch.float)
		# From pre-trained segmentation model, resize
		features = segmentation_pretrained(device, patient_range, cts)
		features = features.cpu().detach().numpy()
		patient_range = [(0,patient_range[i]) if i==0 else (patient_range[i-1], patient_range[i]) for i in range(len(patient_range))]
		features = np.concatenate([np.expand_dims(np.resize(features[patient_range[i][0]:patient_range[i][1]], (data_reader.num_slices,data_reader.f_size*16,int(data_reader.height/16),int(data_reader.width/16))), axis=0) for i in range(len(patient_range))], axis=0)
		features = torch.from_numpy(features).to(device=device, dtype=torch.float)

		for iteration in range(data_reader.classification_iterations):
			classification_model.train()
			# optimizer.zero_grad(set_to_none=True)
			
			with torch.cuda.amp.autocast():
				pred = classification_model(device, patient_range, features)
				# loss = entropy_loss_fn(pred, masks) + dice_loss_fn(pred, masks)