from torch import nn
import torch

class Classification(nn.Module):
	def __init__(self, data_reader):
		super(Classification, self).__init__()
		# self.f_size = f_size
		self.dense1 = nn.Sequential(
			nn.Linear(in_features=data_reader.num_slices*data_reader.f_size*16*int(data_reader.height/16)*int(data_reader.width/16), out_features=100)
		)
		self.dense2 = nn.Sequential(
			nn.Linear(in_features=100, out_features=3)
		)


	def forward(self, device, patient_range, features):
		features = torch.flatten(features, start_dim=1, end_dim=4)
		features = self.dense1(features)
		features = self.dense2(features)
		print(features.shape)

		return features