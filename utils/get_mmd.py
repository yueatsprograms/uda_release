import torch
import numpy as np

def get_centroid(dataloader, model):
	all_outputs = []
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		inputs, labels = inputs.cuda(), labels.cuda()
		with torch.no_grad():
			outputs = model(inputs)
			all_outputs.append(outputs.cpu())
	return torch.mean(torch.cat(all_outputs), dim=0)

def get_mmd(loader_1, loader_2, model):
	model.eval()
	centroid_1 = get_centroid(loader_1, model)
	centroid_2 = get_centroid(loader_2, model)
	model.train()
	return torch.dist(centroid_1, centroid_2, 2).item()

def mmd_select_naive(mmd):
	return np.argmin(mmd)

def mmd_select_scale(mmd, sce):
	sce = np.asarray(sce)
	mmd = np.asarray(mmd)
	scl = np.min(sce) / np.min(mmd)
	return np.argmin(sce + mmd * scl)