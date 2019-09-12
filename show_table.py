from subprocess import call
import sys

args = []
args.append(('mnist_mnistm', 30))
args.append(('mnist_svhn', 90))
args.append(('svhn_mnist', 90))
args.append(('mnist_usps', 30))
args.append(('usps_mnist', 60))
args.append(('cifar_stl', 60))
args.append(('stl_cifar', 60))

for dataset, ylim in args:
	call(' '.join(['python', 'joint_plot.py', 
						dataset, 
						str(ylim)]),
						shell=True)
