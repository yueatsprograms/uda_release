import torch
import torch.utils.data
import numpy as np

class DsetSSQuadRand(torch.utils.data.Dataset):
	# Make sure that your dataset only returns one element!
	def __init__(self, dset):
		self.dset = dset

	def __getitem__(self, index):
		image = self.dset[index]
		label = np.random.randint(4)

		horstr = image.size(1) // 2
		verstr = image.size(2) // 2
		horlab = label // 2
		verlab = label % 2

		image = image[:, horlab*horstr:(horlab+1)*horstr, verlab*verstr:(verlab+1)*verstr,]
		return image, label

	def __len__(self):
		return len(self.dset)
