import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_pano
from data.image_folder import make_dataset
from PIL import Image


class PanoAligned3Dataset(BaseDataset):
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
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w1 = int(256)
        w2 = int(1024)
        A = AB.crop((0, 0, w1, h))
        B = AB.crop((w1, 0, w1+w2, h))
        C = AB.crop((w1+1024, 0, w1+1024+w2, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform_pano(self.opt, transform_params, convert=True, aerial=True)
        transform_params = get_params(self.opt, B.size)
        B_transform = get_transform_pano(self.opt, transform_params, convert=True, aerial=False)
        C_transform = get_transform_pano(self.opt, transform_params, convert=True, aerial=False)

        A = A_transform(A)
        B = B_transform(B)
        D = C_transform(C)


        return {'A': A, 'B': B, 'D': D, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
