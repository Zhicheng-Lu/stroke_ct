import os
import cv2
import numpy as np
import torch
from torch import nn
from data_reader import DataReader
from models.segmentation_pretrained import SegmentationPreTrained
from models.classification_pretrained import ClassificationPreTrained


def severity_train(data_reader, device, time):
	# Load 2 pre-trained models
	segmentation_pretrained = SegmentationPreTrained(data_reader.f_size)
	segmentation_pretrained.load_state_dict(torch.load("checkpoints/segmentation_model.pt"), strict=False)
	segmentation_pretrained = segmentation_pretrained.to(device)

	classification_pretrained = ClassificationPreTrained(data_reader)
	classification_pretrained.load_state_dict(torch.load("checkpoints/classification_model.pt"), strict=False)
	classification_pretrained = classification_pretrained.to(device)

	patient_range, cts, _ = data_reader.read_in_batch('segmentation', 'train')

	