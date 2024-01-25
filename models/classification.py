from torch import nn
import torch

class Classification(nn.Module):
	def __init__(self, data_reader):
		super(Classification, self).__init__()
		self.f_size = self.data_reader.f_size
		self.down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.down4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.down5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.dense1 = nn.Sequential(
			nn.Linear(in_features=data_reader.num_slices*data_reader.f_size*16*int(data_reader.height/16)*int(data_reader.width/16), out_features=50),
			nn.ReLU()
		)
		self.dense2 = nn.Sequential(
			nn.Linear(in_features=50, out_features=50),
			nn.ReLU()
		)
		self.dense3 = nn.Sequential(
			nn.Linear(in_features=50, out_features=3)
		)


	def forward(self, device, cts):
		down1 = self.down1(cts)
		down2 = self.down2(down1)
		down3 = self.down3(down2)
		down4 = self.down4(down3)
		down5 = self.down5(down4)

		fc1 = self.dense1(down5)
		fc2 = self.dense2(fc1)
		output = self.dense(fc2)

		return output