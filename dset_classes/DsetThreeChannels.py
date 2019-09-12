import torch
import torch.utils.data

class DsetThreeChannels(torch.utils.data.Dataset):
	# Make sure that your dataset actually returns images with only one channel!

	def __init__(self, dset):
		self.dset = dset

	def __getitem__(self, index):
		image, label = self.dset[index]
		return image.repeat(3, 1, 1), label

	def __len__(self):
		return len(self.dset)