
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 8):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        
    
        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

    
        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.functional.interpolate(scale_factor=2, mode='nearest'),

            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        
        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        
    
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)
        
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
        #out = out.view(-1, self.n_classes)
#         out = self.softmax(out)
        out = nn.functional.softmax(out,dim=1)
        return out, seg_layer
    

# class Modified3DNet(nn.Module):
#     def __init__(self, in_channels, n_classes, base_n_filter = 8):
#         super(Modified3DNet, self).__init__()
#         self.in_channels = in_channels
#         self.n_classes = n_classes
#         self.base_n_filter = base_n_filter

#         self.lrelu = nn.LeakyReLU()
#         self.dropout3d = nn.Dropout3d(p=0.6)
#         self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
# #         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
#         self.softmax = nn.Softmax(dim=1)

#         # Level 1 context pathway
#         self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
#         self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
#         self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        
#         # Level 2 context pathway
#         self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
#         self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

#         # Level 3 context pathway
#         self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
#         self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

    
#         # Level 4 context pathway
#         self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
#         self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

#         # Level 5 context pathway, level 0 localization pathway
#         self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
#         self.inorm3d_c5 = nn.InstanceNorm3d(self.base_n_filter*16)


#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(128 * 2 * 2 * 2, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(256, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, n_classes),
#         )
        
    
# #         self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
# #         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

#     def conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())

#     def norm_lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
# #             nn.functional.interpolate(scale_factor=2, mode='nearest'),

#             # should be feat_in*2 or feat_in
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())

#     def forward(self, x):
        
#         #  Level 1 context pathway
#         out = self.conv3d_c1_1(x)
#         residual_1 = out
#         out = self.lrelu(out)
#         out = self.conv3d_c1_2(out)
#         out = self.dropout3d(out)
#         out = self.lrelu_conv_c1(out)
#         # Element Wise Summation
#         out += residual_1
#         context_1 = self.lrelu(out)
#         out = self.inorm3d_c1(out)
#         out = self.lrelu(out)

#         # Level 2 context pathway
#         out = self.conv3d_c2(out)
#         residual_2 = out
#         out = self.norm_lrelu_conv_c2(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c2(out)
#         out += residual_2
#         out = self.inorm3d_c2(out)
#         out = self.lrelu(out)
#         context_2 = out

#         # Level 3 context pathway
#         out = self.conv3d_c3(out)
#         residual_3 = out
#         out = self.norm_lrelu_conv_c3(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c3(out)
#         out += residual_3
#         out = self.inorm3d_c3(out)
#         out = self.lrelu(out)
#         context_3 = out

        
#         # Level 4 context pathway
#         out = self.conv3d_c4(out)
#         residual_4 = out
#         out = self.norm_lrelu_conv_c4(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c4(out)
#         out += residual_4
#         out = self.inorm3d_c4(out)
#         out = self.lrelu(out)
#         context_4 = out

        
#         # Level 5 context pathway
#         out = self.conv3d_c5(out)
#         residual_5 = out
#         out = self.norm_lrelu_conv_c5(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c5(out)
#         out += residual_5
#         out = self.inorm3d_c5(out)
#         out = self.lrelu(out)
#         context_5 = out
        
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
       
#         out = nn.functional.softmax(out,dim=1)
        
#         return out
    
    
    
# ORIGINALLY USED
    
# class Modified3DNet(nn.Module):
#     def __init__(self, in_channels, n_classes, base_n_filter = 8):
#         super(Modified3DNet, self).__init__()
#         self.in_channels = in_channels
#         self.n_classes = n_classes
#         self.base_n_filter = base_n_filter

#         self.lrelu = nn.LeakyReLU()
#         self.dropout3d = nn.Dropout3d(p=0.35)
#         self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
# #         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
#         self.softmax = nn.Softmax(dim=1)

    
#         self.intra1 = nn.Sequential(
        
#         # Level 1 context pathway
#         nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter),
#         self.lrelu_conv(self.base_n_filter, self.base_n_filter),
#         nn.Dropout3d(0.35),
        
#         )
            
#         self.intra2 = nn.Sequential(
            
#         # Level 2 context pathway
#         nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*2),
#         self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
#         nn.Dropout3d(0.35),
        
#         )
            
#         self.intra3 = nn.Sequential(
            
#         # Level 3 context pathway
#         nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*4),
#         self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4),
#         nn.Dropout3d(0.25),
        
#         )
        
        
#         self.intra4 = nn.Sequential(
        
    
#         # Level 4 context pathway
#         nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False),
#         self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8),
#         nn.InstanceNorm3d(self.base_n_filter*8),
#         nn.Dropout3d(0.35),
        
#         )
        
        
#         self.intra5 = nn.Sequential(
        

#         # Level 5 context pathway, level 0 localization pathway
#         nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*16),
#         self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
#         nn.Dropout3d(0.35),
            
#         )
                
        
        
#         self.intra6 = nn.Sequential(
        

#         # Level 5 context pathway, level 0 localization pathway
#         nn.Conv3d(self.base_n_filter*16, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*16),
#         self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
#         nn.Dropout3d(0.35),
#         )
               
        
        
#         self.features = nn.Sequential(
            
#             nn.Linear(32 * 2 * 6 * 6, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.35),
#             nn.Linear(512, 64),

#         )
        
#         self.classifier = nn.Sequential(
        
#             nn.Dropout(0.35),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, n_classes),
            
#         )
        
        
# #         self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
# #         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

#     def conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())

#     def norm_lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
# #             nn.functional.interpolate(scale_factor=2, mode='nearest'),

#             # should be feat_in*2 or feat_in
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())

#     def forward(self, x):
               
#         x = self.intra1(x)
#         x = self.intra2(x)
#         x = self.intra3(x)
#         x = self.intra4(x)
#         x = self.intra5(x)
#         x = self.intra6(x)
        
#         out = torch.flatten(x, 1)

#         feat = self.features(out)
#         out = self.classifier(feat)
       
#         out = nn.functional.softmax(out,dim=1)
        
#         return out, feat
    
class Modified3DNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 8):
        super(Modified3DNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.35)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

    
        self.intra1 = nn.Sequential(
        
        # Level 1 context pathway
        nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter),
        self.lrelu_conv(self.base_n_filter, self.base_n_filter),
        nn.Dropout3d(0.35),
        
        )
            
        self.intra2 = nn.Sequential(
            
        # Level 2 context pathway
        nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*2),
        self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
        nn.Dropout3d(0.35),
        
        )
            
        self.intra3 = nn.Sequential(
            
        # Level 3 context pathway
        nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*4),
        self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4),
        nn.Dropout3d(0.25),
        
        )
        
        
        self.intra4 = nn.Sequential(
        
    
        # Level 4 context pathway
        nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False),
        self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8),
        nn.InstanceNorm3d(self.base_n_filter*8),
        nn.Dropout3d(0.35),
        
        )
        
        
        self.intra5 = nn.Sequential(
        

        # Level 5 context pathway, level 0 localization pathway
        nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*16),
        self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
        nn.Dropout3d(0.35),
            
        )
                
        
        
        self.intra6 = nn.Sequential(
        

        # Level 5 context pathway, level 0 localization pathway
        nn.Conv3d(self.base_n_filter*16, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*16),
        self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
        nn.Dropout3d(0.35),
        )
               
        
        
        self.features = nn.Sequential(
            
            nn.Linear(32 * 4 * 3 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 64),

        )
        
        self.classifier = nn.Sequential(
        
            nn.Dropout(0.35),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
            
        )
        
        
#         self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
#         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.functional.interpolate(scale_factor=2, mode='nearest'),

            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        
        x = self.intra1(x)
        x = self.intra2(x)
        x = self.intra3(x)
        x = self.intra4(x)
        x = self.intra5(x)
        x = self.intra6(x)
        
        out = torch.flatten(x, 1)

        feat = self.features(out)
        out = self.classifier(feat)
       
        out = nn.functional.softmax(out,dim=1)
        
        return out, feat
    
    
    
    
    
    
    
    
class Modified3DNetDPIN(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 8):
        super(Modified3DNetDPIN, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.25)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

    
        self.intra1 = nn.Sequential(
        
        # Level 1 context pathway
        nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
        self.intra1_pz = nn.InstanceNorm3d(self.base_n_filter)
        self.intra1_cg = nn.InstanceNorm3d(self.base_n_filter)

            
        self.intra2 = nn.Sequential(
            
        self.lrelu_conv(self.base_n_filter, self.base_n_filter),
        nn.Dropout3d(0.25),

        # Level 2 context pathway
        nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
        )
            
        self.intra2_pz = nn.InstanceNorm3d(self.base_n_filter*2)
        self.intra2_cg = nn.InstanceNorm3d(self.base_n_filter*2)

        self.intra3 = nn.Sequential(
        self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
        nn.Dropout3d(0.25),
            
        # Level 3 context pathway
        nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False),
        
        )
        
        self.intra3_pz = nn.InstanceNorm3d(self.base_n_filter*4)
        self.intra3_cg = nn.InstanceNorm3d(self.base_n_filter*4)
       
        self.intra4 = nn.Sequential(
        
        self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4),
        nn.Dropout3d(0.25),
            
    
        # Level 4 context pathway
        nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*8),
        )
        
        
        self.intra4_pz = nn.InstanceNorm3d(self.base_n_filter*8)
        self.intra4_cg = nn.InstanceNorm3d(self.base_n_filter*8)
        
        
        self.intra5 = nn.Sequential(
        

        self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8),
        nn.Dropout3d(0.25),

        # Level 5 context pathway, level 0 localization pathway
        nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),

            
        )

        
        self.intra5_pz = nn.InstanceNorm3d(self.base_n_filter*16)
        self.intra5_cg = nn.InstanceNorm3d(self.base_n_filter*16)
        
        
        self.relu = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
        self.dropout = nn.Dropout3d(0.25),
            
        self.features = nn.Sequential(
            
            nn.Linear(128 * 2 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 32),

        )
        
        self.classifier = nn.Sequential(
        
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
            
        )
        
        
       
#         self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
#         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.functional.interpolate(scale_factor=2, mode='nearest'),

            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x,zone):
               
        x = self.intra1(x)
        
        if zone == "PZ":
            x = self.intra1_pz(x)
        else:
            x = self.intra1_cg(x)
        
       
        x = self.intra2(x)
        
        if zone == "PZ":
            x = self.intra2_pz(x)
        else:
            x = self.intra2_cg(x)
        
        x = self.intra3(x)
        
        if zone == "PZ":
            x = self.intra3_pz(x)
        else:
            x = self.intra3_cg(x)
        
        
        x = self.intra4(x)
        
        if zone == "PZ":
            x = self.intra4_pz(x)
        else:
            x = self.intra4_cg(x)
        
       
        x = self.intra5(x)
        
        if zone == "PZ":
            x = self.intra5_pz(x)
        else:
            x = self.intra5_cg(x)
        
        
        out = torch.flatten(x, 1)

        feat = self.features(out)
        out = self.classifier(feat)
       
        out = nn.functional.softmax(out,dim=1)
        
        return out, feat
    
    
    
    
    
    
    

class Modified3DParallelNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 8):
        super(Modified3DParallelNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.25)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

    
        self.intra1 = nn.Sequential(
        
        # Level 1 context pathway
        nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter),
#         nn.BatchNorm3d(self.base_n_filter),
        self.lrelu_conv(self.base_n_filter, self.base_n_filter),
        nn.Dropout3d(0.25),
        
        )
            
        self.intra2 = nn.Sequential(
            
        # Level 2 context pathway
        nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*2),
#         nn.BatchNorm3d(self.base_n_filter*2),
        self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
        nn.Dropout3d(0.25),
        
        )
            
#         self.intra3 = nn.Sequential(
            
#         # Level 3 context pathway
#         nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*4),
#         self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4),
#         nn.Dropout3d(0.25),
        
#         )
        
#         self.intra4 = nn.Sequential(
        
    
#         # Level 4 context pathway
#         nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False),
#         self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8),
#         nn.InstanceNorm3d(self.base_n_filter*8),
#         nn.Dropout3d(0.25),
        
#         )
        
        
#         self.intra5 = nn.Sequential(
        

#         # Level 5 context pathway, level 0 localization pathway
#         nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*16),
#         self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
#         nn.Dropout3d(0.25),
            
#         )
        
    
        self.peri1 = nn.Sequential(
        
        # Level 1 context pathway
        nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter),
#         nn.BatchNorm3d(self.base_n_filter),
        self.lrelu_conv(self.base_n_filter, self.base_n_filter),
        nn.Dropout3d(0.25),
        
        )
            
        self.peri2 = nn.Sequential(
            
        # Level 2 context pathway
        nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*2),
#         nn.BatchNorm3d(self.base_n_filter*2),
        self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
        nn.Dropout3d(0.25),
        
        )
            
#         self.peri3 = nn.Sequential(
            
#         # Level 3 context pathway
#         nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*4),
#         self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4),
#         nn.Dropout3d(0.25),
        
#         )
        
        
#         self.peri4 = nn.Sequential(
        
    
#         # Level 4 context pathway
#         nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False),
#         self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8),
#         nn.InstanceNorm3d(self.base_n_filter*8),
#         nn.Dropout3d(0.25),
        
#         )
        
        
#         self.peri5 = nn.Sequential(
        

#         # Level 5 context pathway, level 0 localization pathway
#         nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*16),
#         self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
#         nn.Dropout3d(0.25),
            
#         )


        self.both3 = nn.Sequential(
            
        # Level 3 context pathway
        nn.Conv3d(self.base_n_filter*2*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*4),
#         nn.BatchNorm3d(self.base_n_filter*4),
        self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4),
        nn.Dropout3d(0.25),
        
        )



        self.both4 = nn.Sequential(
        
    
        # Level 4 context pathway
        nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False),
        self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8),
        nn.InstanceNorm3d(self.base_n_filter*8),
#         nn.BatchNorm3d(self.base_n_filter*8),

        nn.Dropout3d(0.25),
        
        )
        
        
        self.both5 = nn.Sequential(
        

        # Level 5 context pathway, level 0 localization pathway
        nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(self.base_n_filter*16),
#         nn.BatchNorm3d(self.base_n_filter*16),
        self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
        nn.Dropout3d(0.25),
            
        )



        
        self.features = nn.Sequential(
            
            nn.Linear(128 * 2 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 64),

        )
        
        self.classifier = nn.Sequential(
        
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
            
        )
        
        
      
    
#         self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
#         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
#             nn.BatchNorm3d(feat_out),
            nn.LeakyReLU()
#             nn.ReLU()
        )

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
#             nn.BatchNorm3d(feat_in),
            nn.LeakyReLU(),
#             nn.ReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
#             nn.ReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
#             nn.BatchNorm3d(feat_in),

            nn.LeakyReLU(),
#             nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.functional.interpolate(scale_factor=2, mode='nearest'),

            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
#             nn.BatchNorm3d(feat_out),
            nn.LeakyReLU()
#             nn.ReLU()
        )

    def forward(self, x,y):
        
        x = self.intra1(x)
        x = self.intra2(x)
#         x = self.intra3(x)
#         x = self.intra4(x)
#         x = self.intra5(x)
        
        y = self.peri1(y)
        y = self.peri2(y)
#         y = self.peri3(y)
#         y = self.peri4(y)
#         y = self.peri5(y)
        
    
        out = torch.cat((x,y), 1)
       
    
    
        out = self.both3(out)
        out = self.both4(out)
        out = self.both5(out)
        

        out = torch.flatten(out, 1)
        
    
#         x = torch.flatten(x, 1)
#         y = torch.flatten(y, 1)
        
        feat = self.features(out)
        out = self.classifier(feat)
       
        out = nn.functional.softmax(out,dim=1)
        
        return out,feat
    
    
    
# class Modified3DParallelNet(nn.Module):
#     def __init__(self, in_channels, n_classes, base_n_filter = 8):
#         super(Modified3DParallelNet, self).__init__()
#         self.in_channels = in_channels
#         self.n_classes = n_classes
#         self.base_n_filter = base_n_filter

#         self.lrelu = nn.LeakyReLU()
#         self.dropout3d = nn.Dropout3d(p=0.25)
#         self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
# #         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
#         self.softmax = nn.Softmax(dim=1)

    
#         self.intra1 = nn.Sequential(
        
#         # Level 1 context pathway
#         nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter),
# #         nn.BatchNorm3d(self.base_n_filter),
#         self.lrelu_conv(self.base_n_filter, self.base_n_filter),
#         nn.Dropout3d(0.25),
        
#         )
            
#         self.intra2 = nn.Sequential(
            
#         # Level 2 context pathway
#         nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*2),
# #         nn.BatchNorm3d(self.base_n_filter*2),
#         self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
#         nn.Dropout3d(0.25),
        
#         )
            

    
#         self.peri1 = nn.Sequential(
        
#         # Level 1 context pathway
#         nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter),
# #         nn.BatchNorm3d(self.base_n_filter),
#         self.lrelu_conv(self.base_n_filter, self.base_n_filter),
#         nn.Dropout3d(0.25),
        
#         )
            
#         self.peri2 = nn.Sequential(
            
#         # Level 2 context pathway
#         nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*2),
# #         nn.BatchNorm3d(self.base_n_filter*2),
#         self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
#         nn.Dropout3d(0.25),
        
#         )
            

            
#         self.glob1 = nn.Sequential(
        
#         # Level 1 context pathway
#         nn.Conv3d(self.in_channels+1, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter),
# #         nn.BatchNorm3d(self.base_n_filter),
#         self.lrelu_conv(self.base_n_filter, self.base_n_filter),
#         nn.Dropout3d(0.25),
        
#         )
            
#         self.glob2 = nn.Sequential(
            
#         # Level 2 context pathway
#         nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*2),
# #         nn.BatchNorm3d(self.base_n_filter*2),
#         self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2),
#         nn.Dropout3d(0.25),
        
#         )
            
            
           
#         self.both3 = nn.Sequential(
            
#         # Level 3 context pathway
#         nn.Conv3d(self.base_n_filter*2*3, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*4),
# #         nn.BatchNorm3d(self.base_n_filter*4),
#         self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4),
#         nn.Dropout3d(0.25),
        
#         )



#         self.both4 = nn.Sequential(
        
    
#         # Level 4 context pathway
#         nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False),
#         self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8),
#         nn.InstanceNorm3d(self.base_n_filter*8),
# #         nn.BatchNorm3d(self.base_n_filter*8),

#         nn.Dropout3d(0.25),
        
#         )
        
        
#         self.both5 = nn.Sequential(
        

#         # Level 5 context pathway, level 0 localization pathway
#         nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False),
#         nn.InstanceNorm3d(self.base_n_filter*16),
# #         nn.BatchNorm3d(self.base_n_filter*16),
#         self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16),
#         nn.Dropout3d(0.25),
            
#         )



#         self.classifier = nn.Sequential(
            
#             nn.Linear(128 * 2 * 2 * 2, 512),
# #             nn.ReLU(inplace=True),
#             nn.LeakyReLU(),
#             nn.Dropout(0.25),
#             nn.Linear(512, 128),
#             nn.Dropout(0.25),
# #             nn.ReLU(inplace=True),
#             nn.LeakyReLU(),
#             nn.Linear(128, n_classes),
#         )
        
    
# #         self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
# #         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

#     def conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
# #             nn.BatchNorm3d(feat_out),
#             nn.LeakyReLU()
# #             nn.ReLU()
#         )

#     def norm_lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
# #             nn.BatchNorm3d(feat_in),
#             nn.LeakyReLU(),
# #             nn.ReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.LeakyReLU(),
# #             nn.ReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
# #             nn.BatchNorm3d(feat_in),

#             nn.LeakyReLU(),
# #             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
# #             nn.functional.interpolate(scale_factor=2, mode='nearest'),

#             # should be feat_in*2 or feat_in
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
# #             nn.BatchNorm3d(feat_out),
#             nn.LeakyReLU()
# #             nn.ReLU()
#         )

#     def forward(self, x,y,z):
        
        
#         x = self.intra1(x)
#         x = self.intra2(x)

        
#         y = self.peri1(y)
#         y = self.peri2(y)

    
#         z = self.glob1(z)
#         z = self.glob2(z)
    
#         out = torch.cat((x,y,z), 1)
         
#         out = self.both3(out)
#         out = self.both4(out)
#         out = self.both5(out)
        

#         out = torch.flatten(out, 1)
        
   
#         out = self.classifier(out)
       
#         out = nn.functional.softmax(out,dim=1)
        
#         return out
    
    
    

def DiceLoss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class ProstateDatasetHDF5(Dataset):
    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.tables = h5py.File(fname,'r', libver='latest', swmr=True)
        self.nitems=self.tables['data'].shape[0]
        self.tables.close()
        self.data = None
        self.mask = None
        self.names = None
        self.transforms = transforms
         
    def __getitem__(self, index):
                
        self.tables = h5py.File(self.fname,'r', libver='latest', swmr=True)
        self.data = self.tables['data']
        self.labels = self.tables['mask']
        
        if "names" in self.tables:
            self.names = self.tables['names']

        img = self.data[index,:,:,:][None]
        mask = self.mask[index,:,:,:][None]
        
        if self.names is not None:
            name = self.names[index]
            
        self.tables.close()
        
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask, name

    def __len__(self):
        return self.nitems
    
# class csPCaDatasetHDF5(Dataset):
#     def __init__(self, fname,transforms = None):
#         self.fname=fname
#         self.tables = h5py.File(fname,'r', libver='latest', swmr=True)
#         self.nitems=self.tables['data'].shape[0]
            
#         self.tables.close()
#         self.data = None
#         self.mask = None 
#         self.labels = None 
#         self.names = None
#         self.transforms = transforms
         
#     def __getitem__(self, index):
                
#         self.tables = h5py.File(self.fname,'r', libver='latest', swmr=True)
#         self.data = self.tables['data']
#         self.mask = self.tables['mask']
#         self.labels = self.tables['labels']
        
#         if "names" in self.tables:
#             self.names = self.tables['names']

#         img = self.data[index,:,:,:]
#         label = self.labels[index]
       
#         ls = self.mask[index,1,:,:,:]
        
#         t2 = img[0]
#         adc = img[1]
        
#         t2 = t2*ls
#         adc = adc*ls
        
#         t2[t2 == 0] = np.median(t2[t2!=0])
#         adc[adc == 0] = np.median(adc[adc!=0])
        
#         intra = np.zeros(img.shape)
       
#         intra[0] = t2 
#         intra[1] = adc
        
        
#         ls = self.mask[index,0,:,:,:]
        
#         t2 = img[0]
#         adc = img[1]
        
#         t2 = t2*ls
#         adc = adc*ls
        
#         t2[t2 == 0] = np.median(t2[t2!=0])
#         adc[adc == 0] = np.median(adc[adc!=0])
        
#         peri = np.zeros(img.shape)
       
#         peri[0] = t2 
#         peri[1] = adc
        
#         if self.names is not None:
#             name = self.names[index]
            
#         self.tables.close()
        
#         img = torch.from_numpy(img)
# #         label = torch.from_numpy(label)

#         return (intra,peri), label

#     def __len__(self):
#         return self.nitems
    
class csPCaDatasetHDF5(Dataset):
    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.tables = h5py.File(fname,'r', libver='latest', swmr=True)
        self.nitems=self.tables['intra'].shape[0]
            
        self.tables.close()
        self.intra = None
        self.intramask = None 
        self.peri = None 
        self.perimask = None
        self.labels = None 
        self.zones = None
        self.names = None
        self.transforms = transforms
         
    def __getitem__(self, index):

   
        self.tables = h5py.File(self.fname,'r', libver='latest', swmr=True)
        
        self.intra = self.tables['intra']
        self.intramask = self.tables['intramask']
        self.peri = self.tables['peri']
        self.perimask = self.tables['perimask']
#         self.glob = self.tables['glob']
#         self.globmask = self.tables['globmask']
        
        
        self.labels = self.tables['labels']
        self.zones = self.tables['zones']
        
        if "names" in self.tables:
            self.names = self.tables['names']

        label = self.labels[index]
        zone = self.zones[index]
        zone = zone.decode("utf-8")
        
    
        intra = self.intra[index,:,:,:]
        peri = self.peri[index,:,:,:]
#         glob = self.glob[index,:,:,:]
    
        ls = self.intramask[index,:,:,:]
        lsperi = self.perimask[index,:,:,:]
        
        
#         lsglob = self.globmask[index,0,:,:,:]
#         pm = self.globmask[index,1,:,:,:]
        
        t2 = intra[0]
        adc = intra[1]
        
        t2 = t2*ls
        adc = adc*ls
        
        t2[t2 == 0] = np.median(t2[t2!=0])
        adc[adc == 0] = np.median(adc[adc!=0])
               
        intra[0] = t2 
        intra[1] = adc
        
                
#         t2 = peri[0]
#         adc = peri[1]
        
#         t2 = t2*lsperi
#         adc = adc*lsperi
        
#         t2[t2 == 0] = np.median(t2[t2!=0])
#         adc[adc == 0] = np.median(adc[adc!=0])
               
#         peri[0] = t2 
#         peri[1] = adc
        
       
#         t2 = glob[0]
#         adc = glob[1]
        
#         t2 = t2*pm
#         adc = adc*pm
        
#         t2[t2 == 0] = np.median(t2[t2!=0])
#         adc[adc == 0] = np.median(adc[adc!=0])
               
#         glob[0] = t2 
#         glob[1] = adc
    
    
        if self.names is not None:
            name = self.names[index]
            name = name.decode("utf-8")

            
        self.tables.close()
        
        
#         glob = np.vstack((glob,lsglob[None]))
        
#         intra = np.vstack((intra,ls[None]))
#         peri = np.vstack((peri,lsperi[None]))

        intra = torch.from_numpy(intra)
        peri = torch.from_numpy(peri)
#         glob = torch.from_numpy(glob)

    
        return (intra,peri), (label, zone, name)

    def __len__(self):
        return self.nitems
    

    
    