import torch
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
import torchvision.utils
import srresnet
from torch.cuda.amp import autocast
import time
class predict():
    def __init__(self):
        path = input("请输入要预测的图片的文件地址")
        self.model = srresnet._NetG()
        model_weights = torch.load('model/model_45.pth')
        self.model.load_state_dict(model_weights)
        self.model = self.model.to('cuda:0')
        self.model.eval()
        self.transformers = transforms.Compose([
            transforms.ToTensor(),
        ])
        print(f"The model has {self.count_parameters():,} parameters.")
        self.img_H,self.img_L = self.get_img(path)
        self.save_path = r"./save_test"

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self):
        self.img_L = self.img_L.to('cuda:0')
        begin_time = time.time()
        img_H = self.model(self.img_L)
        finial_time = time.time()
        total_time = finial_time-begin_time
        print(f"推理花费了{total_time:.5f}秒")
        #保存图片
        # 确保它是在CPU上
        image_tensor = img_H.cpu().squeeze(0)
        print(image_tensor.shape)
        # 转换为numpy数组，并调整数值范围到[0, 255]
        image_numpy = image_tensor.detach().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从[3, H, W]转换为[H, W, 3]
        image_numpy = (image_numpy * 255).astype(np.uint8)
        # 使用PIL保存图像
        image = Image.fromarray(image_numpy)
        image.save('./save_test/output_image.png')


    def get_img(self,path):
        img_H = Image.open(path).convert('RGB')
        w,h = img_H.size
        img_L = img_H.resize((w//2,h//2))
        img_H.save("./save_test/origin.png")
        img_L.save("./save_test/LR.png")
        img_H = self.transformers(img_H)
        img_L = self.transformers(img_L)
        return torch.unsqueeze(img_H,0),torch.unsqueeze(img_L,0)

if __name__ == '__main__':
    p = predict()
    p.predict()