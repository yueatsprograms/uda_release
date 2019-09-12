import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchdata

def prepare_dataset(dataset_name, image_size, channels=3, path='/home/yu/datasets/'):
	if dataset_name == 'usps':
		from dset_loaders.usps import USPS
		tr_dataset = USPS(path+'/usps', download=True, train=True, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.2539,), (0.3842,))
											 ]))
		print('USPS train set size: %d' %(len(tr_dataset)))
		te_dataset = USPS(path+'/usps', download=True, train=False, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.2539,), (0.3842,))
											 ]))
		print('USPS test set size: %d' %(len(te_dataset)))
		if channels == 3:
			from dset_classes.DsetThreeChannels import DsetThreeChannels
			tr_dataset = DsetThreeChannels(tr_dataset)
			te_dataset = DsetThreeChannels(te_dataset)

	elif dataset_name == 'mnist':
		tr_dataset = torchvision.datasets.MNIST(path+'/mnist_pytorch', download=True, train=True, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.1307,), (0.3081,))
										 ]))
		te_dataset = torchvision.datasets.MNIST(path+'/mnist_pytorch', download=True, train=False, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.1307,), (0.3081,))
										 ]))
		if channels == 3:
			from dset_classes.DsetThreeChannels import DsetThreeChannels
			tr_dataset = DsetThreeChannels(tr_dataset)
			te_dataset = DsetThreeChannels(te_dataset)

	elif dataset_name == 'mnistm':
		from dset_loaders.mnistm import MNISTM
		tr_dataset = MNISTM(path+'/mnistm', download=True, train=True, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.4639, 0.4676, 0.4199), (0.2534, 0.2380, 0.2618))
											 ]))
		te_dataset = MNISTM(path+'/mnistm', download=True, train=False, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.4639, 0.4676, 0.4199), (0.2534, 0.2380, 0.2618))
											 ]))

	elif dataset_name == 'svhn':
		tr_dataset = torchvision.datasets.SVHN(root=path+'/svhn', split='train', download=True, transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))
                       ]))
		print('SVHN basic set size: %d' %(len(tr_dataset)))
		te_dataset = torchvision.datasets.SVHN(root=path+'/svhn', split='test', download=True, transform=transforms.Compose([
		                            transforms.Resize(image_size),
		                            transforms.ToTensor(),
		                            transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))
		                       ]))

	elif dataset_name == 'svhn_extra':
		tr_dataset_basic = torchvision.datasets.SVHN(root=path+'/svhn', split='train', download=True, transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))
                       ]))
		print('SVHN basic set size: %d' %(len(tr_dataset_basic)))
		tr_dataset_extra = torchvision.datasets.SVHN(root=path+'/svhn', split='extra', download=True, transform=transforms.Compose([
		                            transforms.Resize(image_size),
		                            transforms.ToTensor(),
		                            transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))
		                       ]))
		print('SVHN extra set size: %d' %(len(tr_dataset_extra)))
		tr_dataset = torchdata.ConcatDataset((tr_dataset_basic, tr_dataset_extra))
		te_dataset = torchvision.datasets.SVHN(root=path+'/svhn', split='test', download=True, transform=transforms.Compose([
		                            transforms.Resize(image_size),
		                            transforms.ToTensor(),
		                            transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))
		                       ]))

	elif dataset_name == 'cifar10':
		tr_dataset = torchvision.datasets.CIFAR10(path+'/', train=True, download=True, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))
										 ]))
		te_dataset = torchvision.datasets.CIFAR10(path+'/', train=False, download=True, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))
										 ]))
		from dset_loaders.modify_cifar_stl import modify_cifar
		modify_cifar(tr_dataset)
		modify_cifar(te_dataset)

	elif dataset_name == 'stl10':
		tr_dataset = torchvision.datasets.STL10(path+'/', split='train', download=True, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
										 ]))
		te_dataset = torchvision.datasets.STL10(path+'/', split='test', download=True, transform=transforms.Compose([
													transforms.Resize(image_size),
													transforms.ToTensor(),
													transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
										 ]))

		from dset_loaders.modify_cifar_stl import modify_stl
		modify_stl(tr_dataset)
		modify_stl(te_dataset)

	else:
		raise ValueError('Dataset %s not found!' %(dataset_name))
	return tr_dataset, te_dataset
