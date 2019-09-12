import numpy as np

def prune_ticks_labels(ticks, labels):
	ticks = np.asarray(ticks)
	labels = np.asarray(labels)

	if len(labels) > 15 and len(labels) <= 50:
		idx = np.where(labels % 5 == 0)
		labels = labels[idx]
		ticks = ticks[idx]
	elif len(labels) > 50 and len(labels) <= 100:
		idx = np.where(labels % 10 == 0)
		labels = labels[idx]
		ticks = ticks[idx]
	elif len(labels) > 100:
		idx = np.where(labels % 25 == 0)
		labels = labels[idx]
		ticks = ticks[idx]
	return ticks, labels

def parse_all_epoch_stats(all_epoch_stats, prune=True):
	base_iter_count = 0
	ticks = []
	labels = []
	xs = []
	tg_te_err = []
	sc_te_err = []
	us_te_err = []
	mmd = []

	for epoch, epoch_stats in enumerate(all_epoch_stats):
		for stats in epoch_stats:
			mmd.append(stats[2])
			tg_te_err.append(stats[3])
			sc_te_err.append(stats[4])
			us_te_err.append(stats[5])
			xs.append(base_iter_count + stats[0])
		base_iter_count += stats[1]
		ticks.append(base_iter_count)
		labels.append(epoch+1)

	if prune:
		ticks, labels = prune_ticks_labels(ticks, labels)
	us_te_err = np.asarray(us_te_err)
	return ticks, labels, xs, tg_te_err, sc_te_err, us_te_err, mmd

def plot_all_epoch_stats(all_epoch_stats, outf):
	import matplotlib.pyplot as plt
	ticks, labels, xs, tg_te_err, sc_te_err, us_te_err, mmd = parse_all_epoch_stats(all_epoch_stats)

	mmd = np.asarray(mmd)
	plt.plot(xs, mmd / np.max(mmd)*100, color='k', label='normalized mmd')
	plt.plot(xs, np.asarray(tg_te_err)*100, color='r', label='target')
	plt.plot(xs, np.asarray(sc_te_err)*100, color='b', label='source')

	colors = ['g', 'm', 'c']
	for i in range(us_te_err.shape[1]):
		plt.plot(xs, np.asarray(us_te_err[:,i])*100, color=colors[i], label='self-sup %d' %(i+1))

	plt.xticks(ticks, labels)
	plt.xlabel('epoch')
	plt.ylabel('test error (%)')
	plt.legend()
	plt.savefig('%s/loss.pdf' %(outf))
	plt.close()
