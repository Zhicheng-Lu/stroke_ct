from torch import nn
import torch

class Classification(nn.Module):
	def __init__(self, data_reader):
		super(Classification, self).__init__()
		self.f_size = data_reader.f_size
		self.down = nn.Sequential(
			nn.Conv3d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2,2,2)),

			nn.Conv3d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2,2,2)),

			nn.Conv3d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2,2,2)),

			nn.Conv3d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2,2,2)),

			nn.Conv3d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2,2,2)),
		)
		self.dense1 = nn.Sequential(
			nn.Linear(in_features=1024*8*8, out_features=100),
			nn.ReLU()
		)
		self.dense2 = nn.Sequential(
			nn.Linear(in_features=100, out_features=50),
			nn.ReLU()
		)
		self.dense3 = nn.Sequential(
			nn.Linear(in_features=50, out_features=3)
		)


	def forward(self, device, cts, data_reader):
		cts = torch.moveaxis(cts, 1, 2)

		down = self.down(cts)
		
		down = torch.flatten(down, start_dim=1, end_dim=4)
		output = self.dense1(down)
		output = self.dense2(output)
		output = self.dense3(output)

		return output