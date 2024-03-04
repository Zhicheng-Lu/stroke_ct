from torch import nn
import torch

class Classification(nn.Module):
	def __init__(self, data_reader):
		super(Classification, self).__init__()
		f_size = data_reader.f_size
		self.down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=f_size, out_channels=f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*f_size, out_channels=4*f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*f_size, out_channels=4*f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		# self.down4 = nn.Sequential(
		# 	nn.MaxPool2d(kernel_size=(2,2)),
		# 	nn.Conv2d(in_channels=4*f_size, out_channels=8*f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU(),
		# 	nn.Conv2d(in_channels=8*f_size, out_channels=8*f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU()
		# )
		# self.down5 = nn.Sequential(
		# 	nn.MaxPool2d(kernel_size=(2,2)),
		# 	nn.Conv2d(in_channels=8*f_size, out_channels=16*f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU(),
		# 	nn.Conv2d(in_channels=16*f_size, out_channels=16*f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU()
		# )
		self.conv3d = nn.Sequential(
			nn.Conv3d(in_channels=4*f_size, out_channels=8*f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2,2,2)),
			nn.Conv3d(in_channels=8*f_size, out_channels=16*f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2,2,2))
		)
		self.dense1 = nn.Sequential(
			# nn.Linear(in_features=data_reader.f_size*16*int(data_reader.num_slices/32)*int(data_reader.height/32)*int(data_reader.width/32), out_features=100),
			nn.Linear(in_features=524288, out_features=100),
			nn.ReLU(),
			nn.Dropout(0.2)
		)
		self.dense2 = nn.Sequential(
			nn.Linear(in_features=100, out_features=100),
			nn.ReLU(),
			nn.Dropout(0.2)
		)
		self.dense3 = nn.Sequential(
			nn.Linear(in_features=100, out_features=3)
		)


	def forward(self, device, cts, data_reader):
		cts = torch.flatten(cts, start_dim=0, end_dim=1)

		down1 = self.down1(cts)
		down2 = self.down2(down1)
		down3 = self.down3(down2)

		features = down3
		features = nn.MaxPool2d((2,2))(features)

		shape = features.shape
		features = features.view(int(shape[0]/data_reader.num_slices), data_reader.num_slices, shape[1], shape[2], shape[3])
		features = torch.moveaxis(features, 1, 2)

		features = self.conv3d(features)

		features = torch.flatten(features, start_dim=1, end_dim=4)

		fc1 = self.dense1(features)
		fc2 = self.dense2(fc1)
		fc3 = self.dense3(fc2)

		output = fc3

		return output