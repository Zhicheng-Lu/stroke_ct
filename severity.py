import os
import numpy as np
import torch
import torch
from torch import nn
from data_reader import DataReader
from models.severity import Severity
from kmeans_pytorch import kmeans, kmeans_predict
from scipy.ndimage import zoom




def severity_train(data_reader, device, time):
	# New classification model
	model = Severity(data_reader.f_size)
	model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	model = model.to(device)

	for stroke_type in ['hemorrhagic', 'ischemic']:

		results = []
		infos = []
		labels = [[],[],[],[]]


		for iteration, (cts_path, masks_path) in enumerate(data_reader.severity[stroke_type]):
			# if not cts_path in ['data/severity/Ischemic/AISD/images/0091373', 'data/severity/Ischemic/AISD/images/0091586', 'data/severity/Ischemic/AISD/images/0226194', 'data/severity/Ischemic/AISD/images/0226262', 'data/severity/Hemorrhagic/Seg-CQ500/images/90', 'data/severity/Hemorrhagic/BHSD/images/181', 'data/severity/Hemorrhagic/Seg-CQ500/images/243', 'data/severity/Hemorrhagic/BHSD/images/110']:
			# 	continue

			cts, masks = data_reader.read_in_batch_severity(cts_path, masks_path)

			cts = torch.from_numpy(cts).to(device=device, dtype=torch.float)

			with torch.no_grad():
				pred = model(device, cts)
				pred = torch.argmax(pred, dim=1)
				# pred = torch.flatten(pred)
				pred = pred.cpu().detach().numpy()
				pred = zoom(pred, (8/pred.shape[0], 50/pred.shape[1], 50/pred.shape[2]))
				results.append(torch.from_numpy(pred))

			infos.append((cts_path, 1000 * np.sum(masks) / (masks.shape[0] * masks.shape[1] * masks.shape[2])))

		results = torch.stack(results)
		results = torch.flatten(results, start_dim=1, end_dim=3)
		
		# kmeans
		cluster_ids_x, cluster_centers = kmeans(
			X=results, num_clusters=4, distance='euclidean', device=torch.device('cuda:0')
		)

		torch.save(cluster_centers, f'checkpoints/severity_{stroke_type}.pt')