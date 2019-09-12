import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.misc import *
from utils.plot_all_epoch_stats import parse_all_epoch_stats
from utils.get_mmd import mmd_select_naive, mmd_select_scale

try:
	all_epoch_stats = torch.load('output/%s/loss.pth' %(sys.argv[1]))
	_, _, _, tg_te_err, _, _, _ = parse_all_epoch_stats(all_epoch_stats)
	print(sys.argv[1] + '\t source only accuracy: %.2f' %((1-tg_te_err[-1])*100))
except:
	print('File does not exist: %s' %(sys.argv[1]))

save_name = 'plots/%s.pdf' %(sys.argv[1])
ylim_upper = int(sys.argv[2])
my_makedir('plots/')

if len(sys.argv) > 3:
	align = int(sys.argv[3])
else:
	align = 1

fnames = []
fnames.append('output/%s_r/loss.pth' %(sys.argv[1]))
if sys.argv[1] == 'svhn_mnist':
	fnames.append('output/%s_rf/loss.pth' %(sys.argv[1]))
else:
	fnames.append('output/%s_rq/loss.pth' %(sys.argv[1]))
	fnames.append('output/%s_rqf/loss.pth' %(sys.argv[1]))

labels = []
labels.append('rot')
if sys.argv[1] == 'svhn_mnist':
	labels.append('rot + flip')
else:
	labels.append('rot + loc')
	labels.append('rot + loc + flip')

colors = ['b', 'r', 'm']
for (fname, label, color) in zip(fnames, labels, colors):
	try:
		all_epoch_stats = torch.load(fname)
	except:
		print('File does not exist: %s' %(fname))
		continue
	ticks, epochs, xs, tg_te_err, sc_te_err, _, mmd = parse_all_epoch_stats(all_epoch_stats)
	mss_error = tg_te_err[mmd_select_scale(mmd, sc_te_err)]

	print(fname + '\t best accuracy: %.2f' %((1-np.min(tg_te_err))*100))
	print(fname + '\t mmd select accuracy: %.2f' %((1-mss_error)*100))

	plt.plot(xs, np.asarray(tg_te_err)*100, color=color, label=label)
	plt.plot(xs, np.asarray(mmd)*align, linestyle='--', color=color)

plt.ylim((0, ylim_upper))
plt.xticks(ticks, epochs)
plt.xlabel('epoch')
plt.ylabel('target test error (%)')
plt.legend()
plt.savefig(save_name)
plt.close()
