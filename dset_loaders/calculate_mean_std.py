import sys
import torchvision
import torchvision.transforms as transforms
import numpy as np

dset_name = sys.argv[1]
dset_dir = sys.argv[2]

if dset_name == "cifar10":
    dataset = torchvision.datasets.CIFAR10(root=dset_dir, train=True)
    print(dataset.train_data.shape)
    print(dataset.train_data.mean(axis=(0,1,2))/255)
    print(dataset.train_data.std(axis=(0,1,2))/255)

elif dset_name == "cifar100":
    dataset = torchvision.datasets.CIFAR100(root=dset_dir, train=True)
    print(dataset.train_data.shape)
    print(dataset.train_data.mean(axis=(0,1,2))/255)
    print(dataset.train_data.std(axis=(0,1,2))/255)

elif dset_name == 'stl10':
    dataset = torchvision.datasets.STL10(root=dset_dir, download=True, split='train')
    print(dataset.data.shape)
    print(dataset.data.mean(axis=(0,2,3))/255)
    print(dataset.data.std(axis=(0,2,3))/255)

elif dset_name == "mnist":    
    dataset = torchvision.datasets.MNIST(root=dset_dir, train=True)
    print(list(dataset.train_data.size()))
    print(dataset.train_data.float().mean()/255)
    print(dataset.train_data.float().std()/255)

elif dset_name == "mnistm":
    from mnistm import MNISTM
    dataset = MNISTM(root=dset_dir, train=True)
    print(list(dataset.train_data.size()))
    for dim in range(3):
        print(dim)
        print(dataset.train_data[:,:,:,dim].float().mean()/255)
        print(dataset.train_data[:,:,:,dim].float().std()/255)

elif dset_name == "svhn":
    dataset = torchvision.datasets.SVHN(root=dset_dir, download=True, split='train')
    print(dataset.data.shape)
    print(dataset.data.mean(axis=(0,2,3))/255)
    print(dataset.data.std(axis=(0,2,3))/255)

elif dset_name == "usps":
    from usps import USPS
    dataset = USPS(root=dset_dir, train=True, download=True)
    dataset = np.asarray([np.asarray(img) for img in dataset.images])
    print(dataset.shape)
    print(dataset.mean()/255)
    print(dataset.std()/255)
