import torch
from .base_model import BaseModel
from . import networks
import numpy as np

class panoganBaselineMOdel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='panoganBaseline', dataset_mode='panoaligned5')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1_seg', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--weight_whole', type=float, default=1, help='weight for the loss of whole image')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.loss_names = ['G_GAN', 'G_GAN_P1', 'G_GAN_P2', 'G_GAN_P3', 'G_GAN_P4',
        self.loss_names = ['D', 'D_img_fake', 'D_img_real', 'D_seg_fake', 'D_seg_real',
                           'G', 'GAN_img_fake', 'GAN_seg_fake', 'L1_img', 'L1_seg']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'real_D', 'fake_D']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'G3', 'G4', 'D1', 'D2', 'D3', 'D4', 'D1_seg', 'D2_seg', 'D3_seg', 'D4_seg']
        else:  # during test time, only load G
            self.model_names = ['G1', 'G2', 'G3', 'G4']
        # define networks (both generator and discriminator)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc * 2, opt.ngf, opt.netG, opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc * 2, opt.ngf, opt.netG, opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG3 = networks.define_G(opt.input_nc, opt.output_nc * 2, opt.ngf, opt.netG, opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG4 = networks.define_G(opt.input_nc, opt.output_nc * 2, opt.ngf, opt.netG, opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD4 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD1_seg = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2_seg = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD3_seg = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD4_seg = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G3 = torch.optim.Adam(self.netG3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G4 = torch.optim.Adam(self.netG4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D3 = torch.optim.Adam(self.netD3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D4 = torch.optim.Adam(self.netD4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D1_seg = torch.optim.Adam(self.netD1_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2_seg = torch.optim.Adam(self.netD2_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D3_seg = torch.optim.Adam(self.netD3_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D4_seg = torch.optim.Adam(self.netD4_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_G3)
            self.optimizers.append(self.optimizer_G4)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)
            self.optimizers.append(self.optimizer_D3)
            self.optimizers.append(self.optimizer_D4)
            self.optimizers.append(self.optimizer_D1_seg)
            self.optimizers.append(self.optimizer_D2_seg)
            self.optimizers.append(self.optimizer_D3_seg)
            self.optimizers.append(self.optimizer_D4_seg)
            self.real_label = 0.9
            self.false_label = 0.0

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_D = input['D' if AtoB else 'C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        cur_input = self.real_A[:, :, :, 256 * 0:256 * 1]
        self.fake_B1, self.fake_D1 = self.netG1(cur_input)  # G(A)

        cur_input = torch.cat((self.fake_B1, self.real_A[:, :, :, 256 * 1:256 * 2]),3)
        self.fake_B12, self.fake_D12 = self.netG2(cur_input)  # G(A)

        cur_input = torch.cat((self.fake_B12, self.real_A[:, :, :, 256 * 2:256 * 3]), 3)
        self.fake_B123, self.fake_D123 = self.netG3(cur_input)  # G(A)

        cur_input = torch.cat((self.fake_B123, self.real_A[:, :, :, 256 * 3:256 * 4]), 3)
        self.fake_B, self.fake_D = self.netG4(cur_input)  # G(A)XS

        if not self.isTrain:
            cur_input = self.real_A[:, :, :, 256 * 3:256 * 4]
            self.fake_B4, self.fake_D4 = self.netG1(cur_input)  # G(A)

            cur_input = torch.cat((self.fake_B4, self.real_A[:, :, :, 256 * 2:256 * 3]), 3)
            self.fake_B43, self.fake_D43 = self.netG2(cur_input)  # G(A)

            cur_input = torch.cat((self.fake_B43, self.real_A[:, :, :, 256 * 1:256 * 2]), 3)
            self.fake_B432, self.fake_D432 = self.netG3(cur_input)  # G(A)

            cur_input = torch.cat((self.fake_B432, self.real_A[:, :, :, 256 * 0:256 * 1]), 3)
            self.fake_B4321, self.fake_D4321 = self.netG4(cur_input)  # G(A)XS

            self.fake_B = (self.fake_B + self.fake_B4321) * 0.5
            self.fake_D = (self.fake_D + self.fake_D4321) * 0.5

        #--------------for patch loss ---------------------------
        begin_D1 = np.random.randint(low=0, high=1024 - 256 * 1, size=1)[0]
        self.idx_D1 = [i for i in range(begin_D1, begin_D1 + 256 * 1)]
        begin_D2 = np.random.randint(low=0, high=1024 - 256 * 2, size=1)[0]
        self.idx_D2 = [i for i in range(begin_D2, begin_D2 + 256 * 2)]
        begin_D3 = np.random.randint(low=0, high=1024 - 256 * 3, size=1)[0]
        self.idx_D3 = [i for i in range(begin_D3, begin_D3 + 256 * 3)]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # --------D1 --------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 1]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 1]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 1]), 1)
        pred_fake = self.netD1(fake_SB.detach())
        self.loss_D_img_fake_1 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD1(real_SB)
        self.loss_D_img_real_1 = self.criterionGAN(pred_real, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 1], real_region_B), 1)
        pred_fake = self.netD1_seg(fake_SB.detach())
        self.loss_D_seg_fake_1 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD1_seg(real_SB)
        self.loss_D_seg_real_1 = self.criterionGAN(pred_real, self.real_label)

        # --------D2--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 2]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 2]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 2]), 1)
        pred_fake = self.netD2(fake_SB.detach())
        self.loss_D_img_fake_12 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD2(real_SB)
        self.loss_D_img_real_12 = self.criterionGAN(pred_real, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 2], real_region_B), 1)
        pred_fake = self.netD2_seg(fake_SB.detach())
        self.loss_D_seg_fake_12 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD2_seg(real_SB)
        self.loss_D_seg_real_12 = self.criterionGAN(pred_real, self.real_label)

        # --------D3--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 3]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 3]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 3]), 1)
        pred_fake = self.netD3(fake_SB.detach())
        self.loss_D_img_fake_123 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD3(real_SB)
        self.loss_D_img_real_123 = self.criterionGAN(pred_real, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 3], real_region_B), 1)
        pred_fake = self.netD3_seg(fake_SB.detach())
        self.loss_D_seg_fake_123 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD3_seg(real_SB)
        self.loss_D_seg_real_123 = self.criterionGAN(pred_real, self.real_label)

        # --------D4--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 4]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 4]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 4]), 1)
        pred_fake = self.netD4(fake_SB.detach())
        self.loss_D_img_fake_1234 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD4(real_SB)
        self.loss_D_img_real_1234 = self.criterionGAN(pred_real, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 4], real_region_B), 1)
        pred_fake = self.netD4_seg(fake_SB.detach())
        self.loss_D_seg_fake_1234 = self.criterionGAN(pred_fake, self.false_label)
        real_SB = torch.cat((real_region_D, real_region_B), 1)
        pred_real = self.netD4_seg(real_SB)
        self.loss_D_seg_real_1234 = self.criterionGAN(pred_real, self.real_label)

        self.loss_D_img_fake = (
                self.loss_D_img_fake_1 + self.loss_D_img_fake_12 +
                self.loss_D_img_fake_123 + self.loss_D_img_fake_1234)

        self.loss_D_img_real = (
                self.loss_D_img_real_1 + self.loss_D_img_real_12 +
                self.loss_D_img_real_123 + self.loss_D_img_real_1234)

        self.loss_D_seg_fake = (
                self.loss_D_seg_fake_1 + self.loss_D_seg_fake_12 +
                self.loss_D_seg_fake_123 + self.loss_D_seg_fake_1234)

        self.loss_D_seg_real = (
                self.loss_D_seg_real_1 + self.loss_D_seg_real_12 +
                self.loss_D_seg_real_123 + self.loss_D_seg_real_1234)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_img_fake + self.loss_D_img_real + self.loss_D_seg_fake + self.loss_D_seg_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # --------G1 --------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 1]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 1]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 1]), 1)
        pred_fake = self.netD1(fake_SB)
        self.loss_GAN_img_fake_1 = self.criterionGAN(pred_fake, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 1], real_region_B), 1)
        pred_fake = self.netD1_seg(fake_SB)
        self.loss_GAN_seg_fake_1 = self.criterionGAN(pred_fake, self.real_label)

        # --------G2--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 2]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 2]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 2]), 1)
        pred_fake = self.netD2(fake_SB)
        self.loss_GAN_img_fake_12 = self.criterionGAN(pred_fake, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 2], real_region_B), 1)
        pred_fake = self.netD2_seg(fake_SB)
        self.loss_GAN_seg_fake_12 = self.criterionGAN(pred_fake, self.real_label)

        # --------D3--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 3]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 3]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 3]), 1)
        pred_fake = self.netD3(fake_SB)
        self.loss_GAN_img_fake_123 = self.criterionGAN(pred_fake, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 3], real_region_B), 1)
        pred_fake = self.netD3_seg(fake_SB)
        self.loss_GAN_seg_fake_123 = self.criterionGAN(pred_fake, self.real_label)

        # --------D4--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 4]
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 4]
        fake_SB = torch.cat((real_region_D, self.fake_B[:, :, :, 256 * 0: 256 * 4]), 1)
        pred_fake = self.netD4(fake_SB)
        self.loss_GAN_img_fake_1234 = self.criterionGAN(pred_fake, self.real_label)
        fake_SB = torch.cat((self.fake_D[:, :, :, 256 * 0: 256 * 4], real_region_B), 1)
        pred_fake = self.netD4_seg(fake_SB)
        self.loss_GAN_seg_fake_1234 = self.criterionGAN(pred_fake, self.real_label)

        # ---------------- total GAN loss -----------------------------------
        self.loss_GAN_img_fake = (
                self.loss_GAN_img_fake_1 + self.loss_GAN_img_fake_12 +
                self.loss_GAN_img_fake_123 + self.loss_GAN_img_fake_1234)

        self.loss_GAN_seg_fake = (
            self.loss_GAN_seg_fake_1 + self.loss_GAN_seg_fake_12 +
            self.loss_GAN_seg_fake_123 + self.loss_GAN_seg_fake_1234)


        # Second, G(A) = B
        # ---------------- Left ==> right -----------------------------------
        self.loss_L1_img1 = self.criterionL1(self.fake_B[:, :, :, 256 * 0: 256 * 1], self.real_B[:, :, :, 256 * 0:256 * 1]) * self.opt.lambda_L1
        self.loss_L1_img12 = self.criterionL1(self.fake_B[:, :, :, 256 * 0: 256 * 2], self.real_B[:, :, :, 256 * 0:256 * 2]) * self.opt.lambda_L1
        self.loss_L1_img123 = self.criterionL1(self.fake_B[:, :, :, 256 * 0: 256 * 3], self.real_B[:, :, :, 256 * 0:256 * 3]) * self.opt.lambda_L1
        self.loss_L1_img1234 = self.criterionL1(self.fake_B[:, :, :, 256 * 0: 256 * 4], self.real_B) * self.opt.lambda_L1

        self.loss_L1_seg1 = self.criterionL1(self.fake_D[:, :, :, 256 * 0: 256 * 1], self.real_D[:, :, :, 256 * 0:256 * 1]) * self.opt.lambda_L1_seg
        self.loss_L1_seg12 = self.criterionL1(self.fake_D[:, :, :, 256 * 0: 256 * 2], self.real_D[:, :, :, 256 * 0:256 * 2]) * self.opt.lambda_L1_seg
        self.loss_L1_seg123 = self.criterionL1(self.fake_D[:, :, :, 256 * 0: 256 * 3], self.real_D[:, :, :, 256 * 0:256 * 3]) * self.opt.lambda_L1_seg
        self.loss_L1_seg1234 = self.criterionL1(self.fake_D[:, :, :, 256 * 0: 256 * 4], self.real_D) * self.opt.lambda_L1_seg

        self.loss_L1_img = self.loss_L1_img1 + self.loss_L1_img12 + self.loss_L1_img123 + self.loss_L1_img1234
        self.loss_L1_seg = self.loss_L1_seg1 + self.loss_L1_seg12 + self.loss_L1_seg123 + self.loss_L1_seg1234

        self.loss_G = self.loss_GAN_img_fake + self.loss_GAN_seg_fake + self.loss_L1_img + self.loss_L1_seg

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD1, True)  # enable backprop for D
        self.set_requires_grad(self.netD2, True)  # enable backprop for D
        self.set_requires_grad(self.netD3, True)  # enable backprop for D
        self.set_requires_grad(self.netD4, True)  # enable backprop for D
        self.set_requires_grad(self.netD1_seg, True)  # enable backprop for D
        self.set_requires_grad(self.netD2_seg, True)  # enable backprop for D
        self.set_requires_grad(self.netD3_seg, True)  # enable backprop for D
        self.set_requires_grad(self.netD4_seg, True)  # enable backprop for D
        self.optimizer_D1.zero_grad()  # set D's gradients to zero
        self.optimizer_D2.zero_grad()  # set D's gradients to zero
        self.optimizer_D3.zero_grad()  # set D's gradients to zero
        self.optimizer_D4.zero_grad()  # set D's gradients to zero
        self.optimizer_D1_seg.zero_grad()  # set D's gradients to zero
        self.optimizer_D2_seg.zero_grad()  # set D's gradients to zero
        self.optimizer_D3_seg.zero_grad()  # set D's gradients to zero
        self.optimizer_D4_seg.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D1.step()  # update D's weights
        self.optimizer_D2.step()  # update D's weights
        self.optimizer_D3.step()  # update D's weights
        self.optimizer_D4.step()  # update D's weights
        self.optimizer_D1_seg.step()  # update D's weights
        self.optimizer_D2_seg.step()  # update D's weights
        self.optimizer_D3_seg.step()  # update D's weights
        self.optimizer_D4_seg.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD1, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD2, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD3, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD4, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD1_seg, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD2_seg, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD3_seg, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD4_seg, False)  # D requires no gradients when optimizing G

        self.optimizer_G1.zero_grad()  # set G's gradients to zero
        self.optimizer_G2.zero_grad()  # set G's gradients to zero
        self.optimizer_G3.zero_grad()  # set G's gradients to zero
        self.optimizer_G4.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G1.step()  # udpate G's weights
        self.optimizer_G2.step()  # udpate G's weights
        self.optimizer_G3.step()  # udpate G's weights
        self.optimizer_G4.step()  # udpate G's weights