import torch
import torch.nn as nn
from piq import multi_scale_ssim

class MSSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(MSSIMLoss, self).__init__()
        self.data_range = data_range

    def forward(self, img1, img2):
        # 计算 MS-SSIM
        ms_ssim_value = multi_scale_ssim(img1, img2, data_range=self.data_range)
        # 计算损失
        loss = 1.0 - ms_ssim_value
        return loss
