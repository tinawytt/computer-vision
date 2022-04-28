#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import nn
# import nn.functional as F

class UNet3DPlus(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=3, norm_layer=nn.InstanceNorm3d):
        # norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNet3DPlus, self).__init__()

        # construct unet structure
        self.unet_block_innermost = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** 2, out_channels=initial_filter_size * 2 ** 3,
                                             num_classes=num_classes, kernel_size=kernel_size, norm_layer=norm_layer, innermost=True)
        # for i in range(1, num_downs):
        #     unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
        #                                          out_channels=initial_filter_size * 2 ** (num_downs-i),
        #                                          num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer)
        self.unet_block_layer3 = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** 1,#128
                                                 out_channels=initial_filter_size * 2 ** 2,#256
                                                 num_classes=num_classes, kernel_size=kernel_size, submodule=self.unet_block_innermost, norm_layer=norm_layer)
        self.unet_block_layer2 = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** 0,#64
                                                 out_channels=initial_filter_size * 2 ** 1,#128
                                                 num_classes=num_classes, kernel_size=kernel_size, submodule=self.unet_block_layer3, norm_layer=norm_layer)
        self.unet_block_outermost = UnetSkipConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                             num_classes=num_classes, kernel_size=kernel_size, submodule=self.unet_block_layer2, norm_layer=norm_layer,
                                             outermost=True)
        self.outconv2=nn.Conv3d(128, num_classes, kernel_size=1) 
        self.outconv3=nn.Conv3d(256, num_classes, kernel_size=1)
        self.outconv4=nn.Conv3d(512, num_classes, kernel_size=1)
        
        self.model = self.unet_block_outermost

    def forward(self, x):
        tuple=self.model(x)
        layer4_output=tuple[0]
        layer3_output=tuple[1]
        layer2_output=tuple[2]
        layer2_after=nn.functional.interpolate(self.outconv2(layer2_output),scale_factor=2, mode='trilinear')
        layer3_after=nn.functional.interpolate(self.outconv3(layer3_output),scale_factor=4,mode='trilinear')
        layer4_after=nn.functional.interpolate(self.outconv4(layer4_output),scale_factor=8,mode='trilinear')
        layer1_output=tuple[-1]
        # return (nn.functional.sigmoid(layer1_output),nn.functional.sigmoid(layer2_after),nn.functional.sigmoid(layer3_after),nn.functional.sigmoid(layer4_after))
        return (layer1_output,layer2_after,layer3_after,layer4_after)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost=innermost
        self.use_dropout=use_dropout
        # downconv
        pool = nn.MaxPool3d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)

        # upconv
        conv3 = self.expand(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        if outermost:
            self.final = nn.Conv3d(out_channels, num_classes, kernel_size=1)
            self.down = nn.Sequential(*[conv1, conv2])
            self.conv = nn.Sequential(*[conv3, conv4, self.final])
            self.sub_module=nn.Sequential(*[submodule])
            
        elif innermost:
            upconv = nn.ConvTranspose3d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            self.down = nn.Sequential(*[pool,conv1, conv2])
            self.up = nn.Sequential(*[upconv])
            self.sub_module=None
        else:
            upconv = nn.ConvTranspose3d(in_channels*2, in_channels, kernel_size=2, stride=2)
            self.down = nn.Sequential(*[pool,conv1, conv2])
            self.sub_module=nn.Sequential(*[submodule])
            self.conv=nn.Sequential(*[conv3, conv4])
            self.up = nn.Sequential(*[upconv])
            if use_dropout:
                self.dropout=nn.Sequential(*[nn.Dropout(0.5)])
            else:
                self.dropout=None

        # self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm3d):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_depth, target_width, target_height):
        batch_size, n_channels, layer_depth, layer_width, layer_height = layer.size()
        xy0 = (layer_depth - target_depth) // 2
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy0:(xy0 + target_depth), xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.innermost:
            mid1=self.down(x)
            mid2=self.up(mid1)
            crop = self.center_crop(mid2, x.size()[2], x.size()[3], x.size()[4])
            list=[]
            list.append(mid1)
            list.append(torch.cat([x, crop], 1))
            return tuple(list)
        elif self.outermost:
            mid1=self.down(x)
            mid2=self.sub_module(mid1)
            tuple_ds=mid2[:-1]
            mid3=self.conv(mid2[-1])
            list=[]
            list.append(mid3)
            tuple_mid3=tuple(list)
            return tuple_ds+tuple_mid3
        else:
            mid1=self.down(x)
            mid2=self.sub_module(mid1)
            mid4=self.conv(mid2[-1])
            tuple_ds=mid2[:-1]
            mid3=self.up(mid4)
            if self.use_dropout :
                crop = self.center_crop(self.dropout(mid3), x.size()[2], x.size()[3], x.size()[4])
                list=[]
                list.append(mid4)
                tuple_new=tuple(list)
                # print("tuple_ds",type(tuple_ds))
                # print("tuple_new",type(tuple_new))
                tuple_return=tuple_ds+tuple_new
                list2=[]
                list2.append(torch.cat([x, crop], 1))
                tuple_this=tuple(list2)
                return tuple_return+tuple_this
            else:
                crop = self.center_crop(mid3, x.size()[2], x.size()[3], x.size()[4])
                list=[]
                list.append(mid4)
                tuple_new=tuple(list)
                
                # print("tuple_ds",type(tuple_ds))
                # print("tuple_new",type(tuple_new))
                tuple_return=tuple_ds+tuple_new
                list2=[]
                list2.append(torch.cat([x, crop], 1))
                tuple_this=tuple(list2)
                return tuple_return+tuple_this
            