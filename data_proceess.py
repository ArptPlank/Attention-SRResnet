import multiprocessing
from PIL import Image
import os
from torchvision import transforms
import torch
from tqdm import tqdm
import threading
from torch.utils.data import DataLoader, Subset, TensorDataset ,random_split
from gc import collect
import numpy as np
class data():
    def __init__(self,batch_size,size,thread_size=2):
        self.data = []
        self.target = []
        self.all_num = 0
        self.size = size
        self.condition = threading.Condition()
        self.transformers_HR = transforms.Compose([
            #transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
        ])
        self.transformers_LR = transforms.Compose([
            #transforms.Resize((size[0]//2, size[1]//2)),
            transforms.ToTensor(),
        ])
        data_path = r"./dataset/Urban100/image_SRF_2"
        data_path = r"./dataset/environment"
        #先获取所有文件名
        self.file_list = os.listdir(data_path)
        for i in range(1,2000):
            try:
                # img_HR = Image.open(f"{data_path}/img_{i:03}_SRF_2_HR.png")
                # img_LR = Image.open(f"{data_path}/img_{i:03}_SRF_2_LR.png")
                img_HR = Image.open(f"{data_path}/HR_{i}.jpg")
                img_LR = img_HR.resize((size[1] // 2, size[0] // 2))
                # img_HR, img_LR = self.call(img_HR, img_LR)
                img_HR = self.transformers_HR(img_HR)
                img_LR = self.transformers_LR(img_LR)
                self.data.append(img_LR)
                self.target.append(img_HR)
            except:
                pass
        print("有{}张图片".format(len(self.data)))
        print("数据加载完成，开始处理数据")
        self.data_set = torch.stack(tuple(self.data), dim=0)
        self.target_set = torch.stack(tuple(self.target), dim=0)
        del self.data, self.target
        collect()
        #self.transformers2 = self.count_mean_std(self.data_set)
        #self.data_set = self.transformers2(self.data_set)
        # self.data_set = [self.transformers2(img) for img in self.data_set]
        # self.data_set = torch.stack(self.data_set)
        self.dataset = TensorDataset(self.data_set, self.target_set)
        del self.data_set, self.target_set
        collect()
        # 定义训练集和验证集的大小
        train_size = int(0.9 * len(self.dataset))  # 假设训练集占80%
        val_size = len(self.dataset) - train_size  # 剩余的部分用作验证集
        # 使用random_split分割数据集
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    def get_data(self):
        return self.train_dataset,self.val_dataset

    def count_mean_std(self,data):
        print("开始计算归一化参数")
        nb_samples = 0.
        channel_mean = torch.zeros(3)
        channel_std = torch.zeros(3)
        N, C, H, W = data.shape[:4]
        data = data.view(N, C, -1)
        print(data.shape)
        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N
        channel_mean /= nb_samples
        channel_std /= nb_samples
        print(f"归一化参数计算完成mean为{channel_mean}，std为{channel_std},开始进行归一化操作")
        return transforms.Compose([transforms.Normalize(channel_mean, channel_std)])

    # def call(self, img_hr, img_lr):
    #     # 确定高分辨率图像的裁剪位置
    #     w, h = img_hr.size
    #     left = np.random.randint(0, w - self.size[1])
    #     top = np.random.randint(0, h - self.size[0])
    #     img_hr_cropped = img_hr.crop((left, top, left + self.size[1], top + self.size[0]))
    #     img_lr_cropped = img_lr.crop((left // 2, top // 2, (left + self.size[1]) // 2, (top + self.size[0]) // 2))
    #     #print(img_hr_cropped.size,img_lr_cropped.size)
    #     return img_hr_cropped, img_lr_cropped


if __name__ == "__main__":
    data = data()
