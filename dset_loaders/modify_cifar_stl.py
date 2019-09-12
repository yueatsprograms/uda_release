import numpy as np
import torch
import torchvision.utils as vutils

def modify_cifar(dataset):
	# remove frogs and shift everything after
	dataset.targets = np.asarray(dataset.targets)
	dataset.data = np.delete(dataset.data, np.where(dataset.targets==6)[0], axis=0)
	dataset.targets = np.delete(dataset.targets, np.where(dataset.targets==6)[0], axis=0)
	dataset.targets = np.asarray([x-1 if x>= 7 else x for x in  dataset.targets])

	# exchange automobiles and birds
	idx1 = np.where(dataset.targets == 1)
	idx2 = np.where(dataset.targets == 2)
	dataset.targets[idx1] = 2
	dataset.targets[idx2] = 1

def modify_stl(dataset):
	# remove monkeys and shift everything after	
	dataset.labels = np.asarray(dataset.labels)
	dataset.data = np.delete(dataset.data, np.where(dataset.labels==7)[0], axis=0)
	dataset.labels = np.delete(dataset.labels, np.where(dataset.labels==7)[0], axis=0)
	dataset.labels = np.asarray([x-1 if x>= 8 else x for x in  dataset.labels])
