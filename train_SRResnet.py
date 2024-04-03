import torch
from data_proceess import data
from tqdm import tqdm
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from PIL import Image
import srresnet
import warnings
from torch.serialization import SourceChangeWarning
import multi_scale_ssim
class train():
    def __init__(self):
        if os.path.exists('./logs') and os.path.isdir('./logs'):
            shutil.rmtree('./logs')
        if os.path.exists('./model') and os.path.isdir('./model'):
            shutil.rmtree('./model')
        warnings.filterwarnings("ignore", category=SourceChangeWarning)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 16
        self.size = [320,512]
        self.num_thread = 2
        data_set = data(self.batch_size,self.size,self.num_thread)
        self.train_data, self.val_data = data_set.get_data()
        # 定义模型
        self.model = srresnet._NetG()
        #加载预训练模型
        self.model.load_state_dict(torch.load("./model_srresnet.pth")['model'].state_dict(),strict=False)
        self.model = self.model.to(self.device)
        print(f"The model has {self.count_parameters():,} parameters.")
        print("train on ", self.device)
        # 定义损失函数
        #self.criterion = torch.nn.MSELoss().to(self.device)
        # 定义感知损失
        #self.criterion = VGG.PerceptualLoss().to(self.device)
        # 定义SSIM损失
        #self.criterion = SSIMLoss(data_range=1.).to(self.device)
        # 定义多尺度SSIM损失
        self.criterion = multi_scale_ssim.MSSIMLoss(data_range=1.).to(self.device)
        self.criterion.eval()
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # 按照损失是否减小定义学习率衰减
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=25,verbose=True,min_lr=1e-8,threshold=5e-7)
        self.writer = SummaryWriter(log_dir='logs')

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(self):
        self.pbar = tqdm(range(2000))
        e = 0
        for epoch in self.pbar:
            if epoch % 5 == 0:
                if not os.path.exists('./model'):
                    os.mkdir('./model')
                torch.save(self.model.state_dict(), f'./model/model_{epoch}.pth')
            self.model.train()
            for index,(batch_data,batch_target) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                output = self.model.forward(batch_data)
                loss = self.criterion(output, batch_target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                torch.cuda.empty_cache()
                self.writer.add_scalar(tag="loss",
                                       scalar_value=loss.item(),
                                       global_step=e
                                       )
                #更新进度条，显示当前的loss（六位小数）和学习率
                loss_value = loss.item()
                formatted_loss = "{:.8f}".format(loss_value)
                learning_rate = self.optimizer.param_groups[0]['lr']
                self.pbar.set_postfix({"loss": formatted_loss, "lr": learning_rate})
                e += 1
    def get_available_memory(self,device_id=0):
        device = torch.device(f"cuda:{device_id}")
        prop = torch.cuda.get_device_properties(device)
        cached = torch.cuda.memory_reserved(device) / (1024 * 1024)  # 这里是预留/缓存的显存
        total = prop.total_memory / (1024 * 1024)  # 总显存

        available_memory = total - cached  # 计算可用显存
        return available_memory  # 返回可用显存大小，单位：MB



if __name__ == "__main__":
    train = train()
    train.train()
