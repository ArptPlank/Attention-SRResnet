import random
import time
import requests
import urllib
import threading
from PIL import Image
from io import BytesIO
from tqdm import tqdm
class get_image():
    def __init__(self):
        # 设置最大线程数为10
        self.semaphore = threading.Semaphore(10)
        self.header = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }
        self.min_width = 512
        self.min_height = 320
        self.target_width = 64
        self.target_height = 64
        self.num = 1
        self.target_num = 200
        #搜索关键词
        self.key_word = "自然风景"

    def get_image(self):
        progress_bar = tqdm(total=self.target_num)
        pn = 30  # pn是从第几张图片获取 百度图片下滑时默认一次性显示30张
        m = 1
        last_num = 1
        while True:
            url = 'https://image.baidu.com/search/acjson?'
            #url = "https://www.bing.com/images/search?q=%E8%87%AA%E7%84%B6%E9%A3%8E%E6%99%AF&qpvt=%E8%87%AA%E7%84%B6%E9%A3%8E%E6%99%AF&form=IGRE&first=1"
            param = {
                'tn': 'resultjson_com',
                'logid': '10939983755485265299',
                'ipn': 'rj',
                'ct': '201326592',
                'is': '',
                'fp': 'result',
                'queryWord': self.key_word,
                'cl': '2',
                'lm': '-1',
                'ie': 'utf-8',
                'oe': 'utf-8',
                'adpicid': '',
                'st': '-1',
                'z': '',
                'ic': '',
                'hd': '',
                'latest': '',
                'copyright': '',
                'word': self.key_word,
                's': '',
                'se': '',
                'tab': '',
                'width': '',
                'height': '',
                'face': '0',
                'istype': '2',
                'qc': '',
                'nc': '1',
                'fr': 'ala',
                'expermode': '',
                'force': '',
                'cg': 'girl',
                'pn': pn,
                'rn': '30',
                'gsm': str(hex(pn)),
            }
            page_info = requests.get(url=url, headers=self.header,params=param)
            page_info.encoding = 'utf-8'  # 确保解析的格式是utf-8的
            page_info = page_info.json()  # 转化为json格式在后面可以遍历字典获取其值
            info_list = page_info['data']  # 观察发现data中存在 需要用到的url地址
            del info_list[-1]  # 每一页的图片30张，下标是从 0 开始 29结束 ，那么请求的数据要删除第30个即 29为下标结束点
            img_path_list = []
            for i in info_list:
                img_path_list.append(i['thumbURL'])
            for index in range(len(img_path_list)):
                #print(img_path_list[index])  # 所有的图片的访问地址
                # 创建一个线程，目标函数是 save_images
                t = threading.Thread(target=self.save_image, args=(img_path_list[index],))
                time.sleep(random.randint(5, 10) * 0.1)  # 随机休眠一段时间
                # 启动线程
                t.start()
                if last_num < self.num:
                    progress_bar.update(1)  # 每次循环手动更新进度条
                    last_num = self.num
            pn += 30
            m += 1

    def save_image(self, url):
        try:
            # 首先获取图片的头部数据来检查分辨率
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # 从响应中加载图片的头部，足以获取其尺寸
            image_head = Image.open(BytesIO(response.content))
            width, height = image_head.size

            # 检查图片是否满足最小分辨率要求
            if width >= self.min_width and height >= self.min_height:
                # 如果满足条件，随机裁剪图片
                max_left = width - self.target_width
                max_top = height - self.target_height
                left = random.randint(0, max_left)
                top = random.randint(0, max_top)
                cropped_image = image_head.crop((left, top, left + self.target_width, top + self.target_height))
                # 保存裁剪后的图片
                file_path_HR = f"./dataset/HR_{self.num}.jpg"
                cropped_image.save(file_path_HR)
                half_width = self.target_width // 2
                half_height = self.target_height // 2
                # 使用双线性插值方法下采样
                downsampled_image = cropped_image.resize((half_width, half_height), Image.BILINEAR)
                # 再次使用双线性插值方法上采样回原始分辨率
                resampled_image = downsampled_image.resize((self.target_width, self.target_height), Image.BILINEAR)
                file_path_LR = f"./dataset/LR_{self.num}.jpg"
                resampled_image.save(file_path_LR)

            self.num += 1
            # urllib.request.urlretrieve(url, "./dataset/" + str(n) + '.jpg')
        except:
            pass

if __name__ == '__main__':
    g = get_image()
    g.get_image()
