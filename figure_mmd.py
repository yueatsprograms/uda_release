import numpy as np
import matplotlib.pyplot as plt

import torch
from utils.plot_all_epoch_stats import parse_all_epoch_stats

def plot_figure_mmd(dataset, ylim):
	all_epoch_stats = torch.load('output/%s_r/loss.pth' %(dataset))
	ticks, labels, xs, tg_te_err, sc_te_err, us_te_err, mmd = parse_all_epoch_stats(all_epoch_stats)

	mmd = np.asarray(mmd)
	mmd = mmd / np.max(mmd) * 100
	plt.plot(xs, mmd, color='k', label='Normalized mean distance')
	plt.plot(xs, np.asarray(tg_te_err)*100, color='r', label='Target error (%)')
	plt.plot(xs, np.asarray(sc_te_err)*100, color='b', label='Source error (%)')
	plt.plot(xs, np.asarray(us_te_err[:,0])*100, color='g', label='Rotation prediction error (%)')
	plt.xticks(ticks, labels)
	plt.xlabel('Epoch')
	plt.ylim(ylim)
	plt.legend()
	plt.savefig('output/figure_mmd_%s.pdf' %(dataset), bbox_inches='tight')
	plt.close()

plot_figure_mmd('mnist_mnistm', [0, 30])
plot_figure_mmd('cifar_stl', [0, 60])