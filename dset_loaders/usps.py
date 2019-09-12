# https://github.com/jhoffman/cycada_release/blob/master/cycada/data/usps.py

import gzip
import os.path

from urllib.parse import urljoin
import numpy as np
from PIL import Image
from torch.utils import data

import logging
import requests

logger = logging.getLogger(__name__)
def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)


def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


class USPS(data.Dataset):

    """USPS handwritten digits.
    Homepage: http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    def __init__(self, root, train=True, transform=None, target_transform=None,
            download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
	
        if download:
            self.download()

        if self.train:
            datapath = os.path.join(self.root, self.data_files['train'])
        else:
            datapath = os.path.join(self.root, self.data_files['test'])

        self.images, self.targets = self.read_data(datapath)
    
    def get_path(self, filename):
        return os.path.join(self.root, filename)

    def download(self):
        data_dir = self.root
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                maybe_download(url, path)

    def read_data(self, path):
        images = []
        targets = []
        with gzip.GzipFile(path, 'r') as f:
            for line in f:
                split = line.strip().split()
                label = int(float(split[0]))
                pixels = np.array([(float(x) + 1) / 2 for x in split[1:]]) * 255
                num_pix = 16
                pixels = pixels.reshape(num_pix, num_pix).astype('uint8')
                img = Image.fromarray(pixels, mode='L')
                images.append(img)
                targets.append(label)
        return images, targets

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)
