from torch import nn
from torchvision.transforms import ToTensor
import torch
import numpy as np

class Segmentation(nn.Module):
	def __init__(self):
		super(Segmentation, self).__init__()
		self.f_size = 64
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
		self.up1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.up2 = nn.Sequential(
			nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.up3 = nn.Sequential(
			nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.up4 = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=2, kernel_size=(1,1))
		)
		# self.dense = nn.Sequential(
		# 	nn.Linear(in_features=self.f_size, out_features=2),
		# 	nn.Softmax(dim=3)
		# )
		

	def forward(self, device, patient_range, cts):
		down1 = self.down1(cts)
		down2 = self.down2(down1)
		down3 = self.down3(down2)
		down4 = self.down4(down3)
		down5 = self.down5(down4)
		up1 = self.up1(down5)
		up1_cat = torch.cat((down4, up1), dim=1)
		up2 = self.up2(up1_cat)
		up2_cat = torch.cat((down3, up2), dim=1)
		up3 = self.up3(up2_cat)
		up3_cat = torch.cat((down2, up3), dim=1)
		up4 = self.up4(up3_cat)
		up4_cat = torch.cat((down1, up4), dim=1)
		output = self.conv(up4_cat)
		# conv = torch.moveaxis(conv, 1, 3)
		# output = self.dense(conv)
		# output = torch.moveaxis(output, 3, 1)

		# print(cts.shape)
		# print(down1.shape)
		# print(down2.shape)
		# print(down3.shape)
		# print(down4.shape)
		# print(down5.shape)
		# print(up1.shape)
		# print(up1_cat.shape)
		# print(up2.shape)
		# print(up2_cat.shape)
		# print(up3.shape)
		# print(up3_cat.shape)
		# print(up4.shape)
		# print(up4_cat.shape)
		# print(conv.shape)
		# print(output.shape)

		return output