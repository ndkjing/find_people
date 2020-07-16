from __future__ import print_function
import sys
import os
#在cmd命令行运行下需要添加如下路径
root_dir = r'D:\PythonProject\find_people'
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir,'core'))
sys.path.append(os.path.join(root_dir,'face'))
sys.path.append(os.path.join(root_dir,'weights'))


from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import cv2
import re
import math
import glob
import time
from collections import namedtuple
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.ops import nms as torch_nms

# from face_ssd import build_ssd
from core.face.face_detect_lightdsfd.model_search import Network
from core.face.face_detect_lightdsfd.dataset.config import widerface_640 as cfg
import base_config

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
def singleton(cls):
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

class BaseTransform:
    def __init__(self, mean):
        #self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def base_transform(self, image, mean):
        # x = cv2.resize(image, (size, size)).astype(np.float32)
        x = image.astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    def __call__(self, image, boxes=None, labels=None):
        return self.base_transform(image, self.mean), boxes, labels


@singleton
class ExteactFace:
    def __init__(self):
        self.trained_model = base_config.trained_model
        self.images_json_path = './images_test/'
        self.videos_path_md5_jsonpath = base_config.videos_path_md5_jsonpath
        self.thresh = 0.9
        self.use_cuda = base_config.use_cuda
        self.transform = BaseTransform((104, 117, 123))
        self.model = self.load_model()

    def load_model(self):
        if self.use_cuda and torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")
        FPN_Genotype = namedtuple("FPN_Genotype", "Inter_Layer Out_Layer")
        AutoFPN = FPN_Genotype(
            Inter_Layer=[
                [("sep_conv_3x3", 1), ("conv_1x1", 0)],
                [("sep_conv_3x3", 2), ("sep_conv_3x3", 0), ("conv_1x1", 1)],
                [("sep_conv_3x3", 3), ("sep_conv_3x3", 1), ("conv_1x1", 2)],
                [("sep_conv_3x3", 4), ("sep_conv_3x3", 2), ("conv_1x1", 3)],
                [("sep_conv_3x3", 5), ("sep_conv_3x3", 3), ("conv_1x1", 4)],
                [("sep_conv_3x3", 4), ("conv_1x1", 5)],
            ],
            Out_Layer=[],
        )
        net = Network(
            C=64,
            criterion=None,
            num_classes=2,
            layers=1,
            phase="test",
            search=False,
            # args=args,
            searched_fpn_genotype=AutoFPN,
            searched_cpm_genotype=None,
            fpn_layers=1,
            cpm_layers=1,
            auxiliary_loss=False,
        )

        state_dict = torch.load(self.trained_model, map_location=lambda storage, loc: storage)
        print("Pretrained model loading OK...")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "auxiliary" not in k:
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            else:
                print("Auxiliary loss is used when retraining.")

        net.load_state_dict(new_state_dict)
        if self.use_cuda:
            net.cuda()
        else:
            net.cpu()
        net.eval()
        print("Finished loading model!")
        return net


    def contain_zh(self,word):
        '''
        判断传入字符串是否包含中文
        :param word: 待判断字符串
        :return: True:包含中文  False:不包含中文
        '''
        zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        word = word.encode('utf8').decode()
        match = zh_pattern.search(word)
        return match

    # 提取图片文件夹
    def process_image(self,image_path=None,image=None,show_image=False):
        # 清除无用张量
        torch.cuda.empty_cache()
        assert image_path is None or image is None
        if image_path:
            if self.contain_zh(image_path):
                image_cv2 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            else:
                image_cv2 = cv2.imread(image_path)
        else:
            image_cv2 = image
        # pre-processing
        image = self.preprocess(image_cv2)

        # post-processing
        detections = self.model(image).view(-1, 5)
        # scale each detection back up to the image

        if self.use_cuda:
            scale = torch.Tensor(
                [
                    image_cv2.shape[1],
                    image_cv2.shape[0],
                    image_cv2.shape[1],
                    image_cv2.shape[0]
                ]
            ).cuda()
        else:
            scale = torch.Tensor(
                [
                    image_cv2.shape[1],
                    image_cv2.shape[0],
                    image_cv2.shape[1],
                    image_cv2.shape[0]
                ]
            ).cpu()
        scores = detections[..., 0]
        boxes = detections[..., 1:] * scale

        # filter the boxes whose score is smaller than 0.8
        keep_mask = (scores >= self.thresh) & (boxes[..., -1] > 2.0)
        scores = scores[keep_mask]
        boxes = boxes[keep_mask]
        # print(scores.max())
        h,w = image_cv2.shape[0:2]
        # print(h,w)
        return_images = []
        keep_idx = torch_nms(boxes, scores, iou_threshold=0.4)
        if len(keep_idx) > 0:
            keep_boxes = boxes[keep_idx].cpu().numpy()
            keep_scores = scores[keep_idx].cpu().numpy()
            for box, s in zip(keep_boxes, keep_scores):
                # 放大检测框
                x_length,y_length  = abs(box[0]-box[2]), abs(box[1]-box[3])
                add_x = x_length * (base_config.face_enlarge - 1) / 2
                add_y = y_length * (base_config.face_enlarge - 1) / 2
                box_large = box+[-add_x,-add_y,add_x,add_y]
                if box_large[0]<0:
                    box_large[0]=0
                if box_large[1]<0:
                    box_large[1]=0
                if box_large[2]>w:
                    box_large[2]=w
                if box_large[3]>h:
                    box_large[3]=h
                box = np.array(box, np.int32)
                box_large = np.array(box_large, np.int32)
                crop_face = image_cv2[box_large[1]:box_large[3], box_large[0]:box_large[2], :]
                return_images.append([crop_face, round(s, 2)])

        print("{} faces are detected in .".format(len(keep_idx)))
        if show_image:
            if len(keep_idx) > 0:
                keep_boxes = boxes[keep_idx].cpu().numpy()
                keep_scores = scores[keep_idx].cpu().numpy()
                for box, s in zip(keep_boxes, keep_scores):
                    cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
                    cv2.rectangle(image_cv2, (box[0], box[1] - 20), (box[0] + 80, box[1] - 2), color=(0, 255, 0),
                                  thickness=-1)
                    cv2.putText(image_cv2, "{:.2f}".format(s), (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                self.thresh, color=(255, 255, 255), thickness=2)
            cv2.imshow('image',image_cv2)
            cv2.waitKey(0)
        return return_images

    # 提取图片文件夹
    def process_images_file(self):
        with open(base_config.images_path_md5_jsonpath,'r') as f:
            images_md5 = json.load(f)
        for images_dir, (md5_code,extracted,name) in images_md5.items():
            images_path = glob.glob(os.path.join(images_dir, "*.png")) + \
                               glob.glob(os.path.join(images_dir, "*.jpg")) + \
                               glob.glob(os.path.join(images_dir, "*.webp")) + \
                               glob.glob(os.path.join(images_dir, "*.bmp"))
            write_image_dir = os.path.join(base_config.extract_face_dir,os.path.split(images_dir)[1])
            if not os.path.exists(write_image_dir):
                os.mkdir(write_image_dir)
            i=0
            for image_path in images_path:
                out_images = self.process_image(image_path=image_path)
                for (out_image, face_confi) in out_images:
                    write_image_path = os.path.join(write_image_dir, str(i) + '_' + str(face_confi) + '.jpg')
                    print('write image', write_image_path)
                    i+=1
                    if self.contain_zh(write_image_path):
                        cv2.imencode('.jpg', out_image)[1].tofile(write_image_path)
                    else:
                        cv2.imwrite(write_image_path, out_image)

    ## 在摄像头上检测
    def web_camera(self):
        with torch.no_grad():
            cap = cv2.VideoCapture(0)
            current_frame = 1
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            detect_second = 1.5  # 检测间隔3秒
            interval = fps * detect_second
            patient = 5
            print('fps', fps)
            while True:
                start_time = time.time()
                flag, frame = cap.read()
                if not flag:
                    print('video is over')
                    break
                # 当未检测到人脸时候间隔检测
                if current_frame % interval == 0:
                    print('interval:', interval)
                    current_time = time.time()
                    # print('read video cost time',current_time-start_time)
                    self.process_image(image=frame,show_image=True)

    # 提取视频文件夹
    def process_video(self,reload_all=False):
        with torch.no_grad():
            with open(self.videos_path_md5_jsonpath,'r') as f:
                videos_path_md5 = json.load(f)
            for video_path,(md5_code,have_image,extracted) in tqdm(videos_path_md5.items()):
                if extracted and not reload_all:
                    print('该视频已检测，跳过',video_path)
                    continue
                # 保存图片序列号
                i=1
                write_image_dir = os.path.join(base_config.extract_face_dir, md5_code)
                if not os.path.exists(write_image_dir):
                    os.mkdir(write_image_dir)
                # 处理文件夹中图片
                if have_image:
                    images_name = glob.glob(os.path.join(os.path.split(video_path)[0], "*.png")) + \
                                   glob.glob(os.path.join(os.path.split(video_path)[0], "*.jpg")) + \
                                   glob.glob(os.path.join(os.path.split(video_path)[0], "*.bmp"))
                    if len(images_name) ==0:
                        print('存储有图片的文件夹下没有检测到图片',video_path)
                    else:
                        for image_name in images_name:
                            image_path = os.path.join(os.path.split(video_path)[0],image_name)
                            # 传递给图片处理函数
                            out_images = self.process_image(image_path=image_path)
                            for out_image,face_confi in out_images:
                                # 根据文件夹长度命名
                                write_image_path = os.path.join(write_image_dir, str(i) + '_' + str(face_confi) + '.jpg')
                                cv2.imwrite(write_image_path,out_image)
                                i+=1

                # 处理视频
                print('######################video_path',video_path)
                cap = cv2.VideoCapture(video_path)
                current_frame=1
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                detect_second=1.5   # 检测间隔3秒
                interval = fps*detect_second
                patient = 5
                print('fps', fps)
                while True:
                    start_time = time.time()
                    flag,frame = cap.read()
                    if not flag:
                        print('video is over')
                        break
                    # 当未检测到人脸时候间隔检测
                    if current_frame%interval==0:
                        print('interval:',interval)
                        current_time = time.time()
                        # print('read video cost time',current_time-start_time)
                        out_images = self.process_image(image=frame)
                        detect_time = time.time()
                        print('detect cost time', detect_time - current_time)
                        # 当检测到人脸时减小检测间隔  超过一定间隔未检测到就增大间隔间隔
                        if len(out_images)!=0:
                            interval=int(fps//2)
                            patient=5
                        if interval==int(fps//2) and len(out_images)==0:
                            patient-=1
                        if patient<=0:
                            interval=fps*detect_second

                        for (out_image,face_confi) in out_images:

                            write_image_path = os.path.join(write_image_dir, str(i)+'_'+str(face_confi) + '.jpg')
                            print('write image',write_image_path)
                            cv2.imwrite(write_image_path, out_image)
                            i += 1
                        # print('write time', time.time() - detect_time)
                    current_frame+=1
                    # 处理完视频后 更新视频为已处理标志位
                videos_path_md5.update({video_path: [md5_code, have_image, True]})
                with open(base_config.videos_path_md5_jsonpath, 'w') as f:
                    json.dump(videos_path_md5, f)

    def preprocess(self,img):
        if len(img.shape)==3:
            x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
        elif len(img.shape)==4:
            x = torch.from_numpy(self.transform(img)[0]).permute(0,3, 1, 2)
        else:
            raise Exception('error x ')
        # print(x.shape,len(x.shape))

        if self.use_cuda:
            if len(x.shape)==3:
                x = x.unsqueeze(0).cuda()
            elif len(x.shape) == 4:
                x = x.cuda()
        else:
            x = x.unsqueeze(0).cpu()
        return x




if __name__=='__main__':
    # 测试单张图片检测效果
    obj = ExteactFace()
    # i.process_image(image_path=r'D:\PythonProject\find_people\core\face\face_detect_lightdsfd\images_test\yuebing.jpg',show_image=True)
    # i.process_video(reload_all=False)
    obj.web_camera()
    # i.process_images_file()
