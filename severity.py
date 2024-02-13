import os
# import cv2
import numpy as np
import torch
from torch import nn
from data_reader import DataReader
from models.severity import Severity
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt



def severity_train(data_reader, device, time):
	# New classification model
	model = Severity(data_reader.f_size)
	model.load_state_dict(torch.load("checkpoints/classification_model.pt"), strict=False)
	model = model.to(device)

	for stroke_type in ['hemorrhagic', 'ischemic']:

		results = []
		infos = []
		labels = [[],[],[],[]]


		for iteration, (cts_path, masks_path) in enumerate(data_reader.severity[stroke_type]):
			cts, masks = data_reader.read_in_batch_severity(cts_path, masks_path)

			cts = torch.from_numpy(cts).to(device=device, dtype=torch.float)

			with torch.no_grad():
				pred = model(device, cts)
				pred = torch.argmax(pred, dim=1)
				# pred = torch.flatten(pred)
				pred = pred.cpu().detach().numpy()
				results.append(torch.from_numpy(pred))

			infos.append((cts_path, 1000 * np.sum(masks) / (masks.shape[0] * masks.shape[1] * masks.shape[2])))

			if iteration == 200:
				break

		results = torch.stack(results)
		results = torch.flatten(results, start_dim=1, end_dim=3)
		print(results.shape)
		
		# kmeans
		cluster_ids_x, cluster_centers = kmeans(
			X=results, num_clusters=4, distance='euclidean', device=torch.device('cuda:0')
		)

		# print(cluster_ids_x, cluster_centers)

		for i,cluster_id in enumerate(cluster_ids_x):
			labels[cluster_id].append(infos[i])


		print(labels)
