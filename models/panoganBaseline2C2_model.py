import torch
from .base_model import BaseModel
from . import networks
import numpy as np

"""
panoganBaseline2C2
    1. framework: progressive generation (left ---> right)
    2. backbone: x-seq + unet256
    3. input: 256 x 256 (counter-clockwise rotation for each progressive phrase)
    4. GAN loss for image and seg, input of Discriminator is the concate(previous input, real_img/fake_img, real_seg/fake_seg)
    5. L1 loss for img and seg
    6. feedback adversarial loop (iccv19)
"""
class panoganBaseline2C2MOdel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_dcn', dataset_mode='panoaligned4')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1_seg', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.loss_names = ['G_GAN', 'G_GAN_P1', 'G_GAN_P2', 'G_GAN_P3', 'G_GAN_P4',
        self.loss_names = ['D_img_fake', 'D_img_real', 'D_seg_fake', 'D_seg_real',
                           'GAN_img', 'GAN_seg',
                           'L1_img', 'L1_seg'
                           ]

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'real_D', 'fake_D']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G_img1', 'G_img2', 'G_img3', 'G_img4',
                                'G_seg1', 'G_seg2', 'G_seg3', 'G_seg4',
                                'D_img1', 'D_img2', 'D_img3', 'D_img4',
                                'D_seg1', 'D_seg2', 'D_seg3', 'D_seg4']
        else:  # during test time, only load G
            self.model_names = ['G_img1', 'G_img2', 'G_img3', 'G_img4']
        # define networks (both generator and discriminator)
        self.netD_img1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_img2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_img3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_img4 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_seg1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_seg2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_seg3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_seg4 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_img1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_img1)
        self.netG_img2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_img2)
        self.netG_img3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_img3)
        self.netG_img4 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_img4)
        self.netG_seg1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_seg1)
        self.netG_seg2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_seg2)
        self.netG_seg3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_seg3)
        self.netG_seg4 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.netD_seg4)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc


            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_Gi1 = torch.optim.Adam(self.netG_img1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gi2 = torch.optim.Adam(self.netG_img2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gi3 = torch.optim.Adam(self.netG_img3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gi4 = torch.optim.Adam(self.netG_img4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gs1 = torch.optim.Adam(self.netG_seg1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gs2 = torch.optim.Adam(self.netG_seg2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gs3 = torch.optim.Adam(self.netG_seg3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gs4 = torch.optim.Adam(self.netG_seg4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Di1 = torch.optim.Adam(self.netD_img1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Di2 = torch.optim.Adam(self.netD_img2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Di3 = torch.optim.Adam(self.netD_img3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Di4 = torch.optim.Adam(self.netD_img4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Ds1 = torch.optim.Adam(self.netD_seg1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Ds2 = torch.optim.Adam(self.netD_seg2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Ds3 = torch.optim.Adam(self.netD_seg3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Ds4 = torch.optim.Adam(self.netD_seg4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Gi1)
            self.optimizers.append(self.optimizer_Gi2)
            self.optimizers.append(self.optimizer_Gi3)
            self.optimizers.append(self.optimizer_Gi4)
            self.optimizers.append(self.optimizer_Gs1)
            self.optimizers.append(self.optimizer_Gs2)
            self.optimizers.append(self.optimizer_Gs3)
            self.optimizers.append(self.optimizer_Gs4)
            self.optimizers.append(self.optimizer_Di1)
            self.optimizers.append(self.optimizer_Di2)
            self.optimizers.append(self.optimizer_Di3)
            self.optimizers.append(self.optimizer_Di4)
            self.optimizers.append(self.optimizer_Ds1)
            self.optimizers.append(self.optimizer_Ds2)
            self.optimizers.append(self.optimizer_Ds3)
            self.optimizers.append(self.optimizer_Ds4)

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
        self.input1 = self.real_A[:, :, :, 256 * 0:256 * 1]
        self.fake_B1 = self.netG_img1(self.input1)
        self.fake_D1 = self.netG_seg1(self.fake_B1)

        self.input12 = torch.cat((self.fake_B1, self.real_A[:, :, :, 256 * 1:256 * 2]), 3)
        self.fake_B12 = self.netG_img2(self.input12)
        self.fake_D12 = self.netG_seg2(self.fake_B12)

        self.input123 = torch.cat((self.fake_B12, self.real_A[:, :, :, 256 * 2:256 * 3]), 3)
        self.fake_B123 = self.netG_img3(self.input123)
        self.fake_D123 = self.netG_seg3(self.fake_B123)

        self.input1234 = torch.cat((self.fake_B123, self.real_A[:, :, :, 256 * 3:256 * 4]), 3)
        self.fake_B = self.netG_img4(self.input1234)
        self.fake_D = self.netG_seg4(self.fake_B)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # --------D1 --------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 1]
        fake_AB = torch.cat((self.input1, self.fake_B1), 1)
        pred_fake = self.netD_img1(fake_AB.detach())
        self.loss_D_img_fake_1 = self.criterionGAN(pred_fake, self.false_label)
        real_AB = torch.cat((self.input1, real_region_B), 1)
        pred_real = self.netD_img1(real_AB)
        self.loss_D_img_real_1 = self.criterionGAN(pred_real, self.real_label)

        """
        self.loss_D_seg_real_1 = 0
        self.loss_D_seg_fake_1 = 0
        """
        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 1]
        fake_BD = torch.cat((self.fake_B1, self.fake_D1), 1)
        pred_fake = self.netD_seg1(fake_BD.detach())
        self.loss_D_seg_fake_1 = self.criterionGAN(pred_fake, self.false_label)
        real_BD = torch.cat((self.fake_B1, real_region_D), 1)
        pred_real = self.netD_seg1(real_BD)
        self.loss_D_seg_real_1 = self.criterionGAN(pred_real, self.real_label)


        # --------D2--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 2]
        fake_AB = torch.cat((self.input12, self.fake_B12), 1)
        pred_fake = self.netD_img2(fake_AB.detach())
        self.loss_D_img_fake_12 = self.criterionGAN(pred_fake, self.false_label)
        real_AB = torch.cat((self.input12, real_region_B), 1)
        pred_real = self.netD_img2(real_AB)
        self.loss_D_img_real_12 = self.criterionGAN(pred_real, self.real_label)

        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 2]
        fake_BD = torch.cat((self.fake_B12, self.fake_D12), 1)
        pred_fake = self.netD_seg2(fake_BD.detach())
        self.loss_D_seg_fake_12 = self.criterionGAN(pred_fake, self.false_label)
        real_BD = torch.cat((self.fake_B12, real_region_D), 1)
        pred_real = self.netD_seg2(real_BD)
        self.loss_D_seg_real_12 = self.criterionGAN(pred_real, self.real_label)

        # --------D3--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 3]
        fake_AB = torch.cat((self.input123, self.fake_B123), 1)
        pred_fake = self.netD_img3(fake_AB.detach())
        self.loss_D_img_fake_123 = self.criterionGAN(pred_fake, self.false_label)
        real_AB = torch.cat((self.input123, real_region_B), 1)
        pred_real = self.netD_img3(real_AB)
        self.loss_D_img_real_123 = self.criterionGAN(pred_real, self.real_label)

        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 3]
        fake_BD = torch.cat((self.fake_B123, self.fake_D123), 1)
        pred_fake = self.netD_seg3(fake_BD.detach())
        self.loss_D_seg_fake_123 = self.criterionGAN(pred_fake, self.false_label)
        real_BD = torch.cat((self.fake_B123, real_region_D), 1)
        pred_real = self.netD_seg3(real_BD)
        self.loss_D_seg_real_123 = self.criterionGAN(pred_real, self.real_label)
        # --------D4--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 4]
        fake_AB = torch.cat((self.input1234, self.fake_B), 1)
        pred_fake = self.netD_img4(fake_AB.detach())
        self.loss_D_img_fake_1234 = self.criterionGAN(pred_fake, self.false_label)
        real_AB = torch.cat((self.input1234, real_region_B), 1)
        pred_real = self.netD_img4(real_AB)
        self.loss_D_img_real_1234 = self.criterionGAN(pred_real, self.real_label)

        real_region_D = self.real_D[:, :, :, 256 * 0: 256 * 4]
        fake_BD = torch.cat((self.fake_B, self.fake_D), 1)
        pred_fake = self.netD_seg4(fake_BD.detach())
        self.loss_D_seg_fake_1234 = self.criterionGAN(pred_fake, self.false_label)
        real_BD = torch.cat((self.fake_B, real_region_D), 1)
        pred_real = self.netD_seg4(real_BD)
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

        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # --------G1 --------------------
        fake_AB = torch.cat((self.input1, self.fake_B1), 1)
        pred_fake = self.netD_img1(fake_AB)
        self.loss_GAN_img_1 = self.criterionGAN(pred_fake, self.real_label)

        fake_BD = torch.cat((self.fake_B1, self.fake_D1), 1)
        pred_fake = self.netD_seg1(fake_BD)
        self.loss_GAN_seg_1 = self.criterionGAN(pred_fake, self.real_label)

        # --------D2--------------------
        fake_AB = torch.cat((self.input12, self.fake_B12), 1)
        pred_fake = self.netD_img2(fake_AB)
        self.loss_GAN_img_12 = self.criterionGAN(pred_fake, self.real_label)

        fake_BD = torch.cat((self.fake_B12, self.fake_D12), 1)
        pred_fake = self.netD_seg2(fake_BD)
        self.loss_GAN_seg_12 = self.criterionGAN(pred_fake, self.real_label)

        # --------D3--------------------
        fake_AB = torch.cat((self.input123, self.fake_B123), 1)
        pred_fake = self.netD_img3(fake_AB)
        self.loss_GAN_img_123 = self.criterionGAN(pred_fake, self.real_label)

        fake_BD = torch.cat((self.fake_B123, self.fake_D123), 1)
        pred_fake = self.netD_seg3(fake_BD)
        self.loss_GAN_seg_123 = self.criterionGAN(pred_fake, self.real_label)

        # --------D4--------------------
        real_region_B = self.real_B[:, :, :, 256 * 0: 256 * 4]
        fake_AB = torch.cat((self.input1234, self.fake_B), 1)
        pred_fake = self.netD_img4(fake_AB)
        self.loss_GAN_img_1234 = self.criterionGAN(pred_fake, self.real_label)

        fake_BD = torch.cat((self.fake_B, self.fake_D), 1)
        pred_fake = self.netD_seg4(fake_BD)
        self.loss_GAN_seg_1234 = self.criterionGAN(pred_fake, self.real_label)

        # ---------------- total GAN loss -----------------------------------
        self.loss_GAN_img = (
                self.loss_GAN_img_1 + self.loss_GAN_img_12 +
                self.loss_GAN_img_123 + self.loss_GAN_img_1234)
        self.loss_GAN_seg = (
                self.loss_GAN_seg_1 + self.loss_GAN_seg_12 +
                self.loss_GAN_seg_123 + self.loss_GAN_seg_1234)
        self.loss_GAN = self.loss_GAN_img + self.loss_GAN_seg
        # Second, G(A) = B
        # ---------------- Left ==> right -----------------------------------
        self.loss_L1_img1 = self.criterionL1(self.fake_B1, self.real_B[:, :, :, 256 * 0:256 * 1]) * self.opt.lambda_L1
        self.loss_L1_img12 = self.criterionL1(self.fake_B12, self.real_B[:, :, :, 256 * 0:256 * 2]) * self.opt.lambda_L1
        self.loss_L1_img123 = self.criterionL1(self.fake_B123, self.real_B[:, :, :, 256 * 0:256 * 3]) * self.opt.lambda_L1
        self.loss_L1_img1234 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_L1_seg1 = self.criterionL1(self.fake_D1, self.real_D[:, :, :, 256 * 0:256 * 1]) * self.opt.lambda_L1_seg
        self.loss_L1_seg12 = self.criterionL1(self.fake_D12, self.real_D[:, :, :, 256 * 0:256 * 2]) * self.opt.lambda_L1_seg
        self.loss_L1_seg123 = self.criterionL1(self.fake_D123, self.real_D[:, :, :, 256 * 0:256 * 3]) * self.opt.lambda_L1_seg
        self.loss_L1_seg1234 = self.criterionL1(self.fake_D, self.real_D) * self.opt.lambda_L1_seg

        self.loss_L1_img = self.loss_L1_img1 + self.loss_L1_img12 + self.loss_L1_img123 + self.loss_L1_img1234
        self.loss_L1_seg = self.loss_L1_seg1 + self.loss_L1_seg12 + self.loss_L1_seg123 + self.loss_L1_seg1234

        self.loss_L1 = self.loss_L1_img + self.loss_L1_seg


        self.loss_G = self.loss_GAN + self.loss_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD_img1, True)  # enable backprop for D
        self.set_requires_grad(self.netD_img2, True)  # enable backprop for D
        self.set_requires_grad(self.netD_img3, True)  # enable backprop for D
        self.set_requires_grad(self.netD_img4, True)  # enable backprop for D
        self.set_requires_grad(self.netD_seg1, True)  # enable backprop for D
        self.set_requires_grad(self.netD_seg2, True)  # enable backprop for D
        self.set_requires_grad(self.netD_seg3, True)  # enable backprop for D
        self.set_requires_grad(self.netD_seg4, True)  # enable backprop for D

        self.optimizer_Di1.zero_grad()  # set D's gradients to zero
        self.optimizer_Di2.zero_grad()  # set D's gradients to zero
        self.optimizer_Di3.zero_grad()  # set D's gradients to zero
        self.optimizer_Di4.zero_grad()  # set D's gradients to zero
        self.optimizer_Ds1.zero_grad()  # set D's gradients to zero
        self.optimizer_Ds2.zero_grad()  # set D's gradients to zero
        self.optimizer_Ds3.zero_grad()  # set D's gradients to zero
        self.optimizer_Ds4.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_Di1.step()  # update Di's weights
        self.optimizer_Di2.step()  # update Di's weights
        self.optimizer_Di3.step()  # update Di's weights
        self.optimizer_Di4.step()  # update Ds's weights
        self.optimizer_Ds1.step()  # update Ds's weights
        self.optimizer_Ds2.step()  # update Ds's weights
        self.optimizer_Ds3.step()  # update Ds's weights
        self.optimizer_Ds4.step()  # update Ds's weights
        # update G
        self.set_requires_grad(self.netD_img1, False)  # Di requires no gradients when optimizing G
        self.set_requires_grad(self.netD_img2, False)  # Di requires no gradients when optimizing G
        self.set_requires_grad(self.netD_img3, False)  # Di requires no gradients when optimizing G
        self.set_requires_grad(self.netD_img4, False)  # Di requires no gradients when optimizing G
        self.set_requires_grad(self.netD_seg1, False)  # Ds requires no gradients when optimizing G
        self.set_requires_grad(self.netD_seg2, False)  # Ds requires no gradients when optimizing G
        self.set_requires_grad(self.netD_seg3, False)  # Ds requires no gradients when optimizing G
        self.set_requires_grad(self.netD_seg4, False)  # Ds requires no gradients when optimizing G

        self.optimizer_Gi1.zero_grad()  # set Di's gradients to zero
        self.optimizer_Gi2.zero_grad()  # set Di's gradients to zero
        self.optimizer_Gi3.zero_grad()  # set Di's gradients to zero
        self.optimizer_Gi4.zero_grad()  # set Di's gradients to zero
        self.optimizer_Gs1.zero_grad()  # set Di's gradients to zero
        self.optimizer_Gs2.zero_grad()  # set Ds's gradients to zero
        self.optimizer_Gs3.zero_grad()  # set Ds's gradients to zero
        self.optimizer_Gs4.zero_grad()  # set Ds's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_Gi1.step()  # udpate Di's weights
        self.optimizer_Gi2.step()  # udpate Di's weights
        self.optimizer_Gi3.step()  # udpate Di's weights
        self.optimizer_Gi4.step()  # udpate Di's weights
        self.optimizer_Gs1.step()  # udpate Ds's weights
        self.optimizer_Gs2.step()  # udpate Ds's weights
        self.optimizer_Gs3.step()  # udpate Ds's weights
        self.optimizer_Gs4.step()  # udpate Ds's weights