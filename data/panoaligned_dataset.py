import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import torch

class PanoAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        ABC = Image.open(AB_path).convert('RGB')
        # split ABC image into aerial, pano, pano_seg
        w, h = ABC.size
        w1 = int(256)
        w2 = int(1024)
        A_tmp = ABC.crop((0,0,w1,h))
        #A_tmp = Image.fromarray(ABC.crop((0, 0, w1, h)))
        B = ABC.crop((w1, 0, w1 + w2, h))
        D = ABC.crop((w1 + w2, 0, w1 + w2 + w2, h))
        A = transforms.ToTensor()(A_tmp)
        for i in range(1, 4):
            #A.append(A_tmp.rotate(90*i))
            A = torch.cat((A, transforms.ToTensor()(A_tmp.rotate(90*i))),2)

        #A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        D = transforms.ToTensor()(D)

        #h_offset = random.randint(0, max(0, self.opt.load_size - self.opt.crop_size - 1))
        #A = A[:, h_offset:h_offset + self.opt.crop_size, h_offset:h_offset + self.opt.crop_size]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        D = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(D)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc
        """
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            idx = [i for i in range(B.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            B = B.index_select(2, idx)
            D = D.index_select(2, idx)
        """
        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'D': D, 'C': [],
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
