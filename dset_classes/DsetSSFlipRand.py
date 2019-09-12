import torch
import torch.utils.data
import numpy as np

# Assumes that the input tensor is (nchannels, height, width)
def flip_ver(x):
	return x.flip(1)
def flip_hor(x):
	return x.flip(2)

class DsetSSFlipRand(torch.utils.data.Dataset):
	# Make sure that your dataset only returns one element!
	# Horizontal flips are reasonable for digits, not for natural scenes
	def __init__(self, dset, digit=False):
		self.dset = dset
		if digit:
			self.label_max = 3
		else:
			self.label_max = 2

	def __getitem__(self, index):
		image = self.dset[index]
		label = np.random.randint(self.label_max)

		if label == 1:
			image = flip_ver(image)
		elif label == 2:
			image = flip_hor(image)
		return image, label

	def __len__(self):
		return len(self.dset)
