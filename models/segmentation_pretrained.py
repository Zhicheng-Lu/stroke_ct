from torch import nn
from torchvision.transforms import ToTensor
import torch
import numpy as np

class SegmentationPreTrained(nn.Module):
	def __init__(self, f_size):
		super(SegmentationPreTrained, self).__init__()
		self.f_size = f_size
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

	def forward(self, device, patient_range, cts):
		down1 = self.down1(cts)
		down2 = self.down2(down1)
		down3 = self.down3(down2)
		down4 = self.down4(down3)
		down5 = self.down5(down4)

		return down5