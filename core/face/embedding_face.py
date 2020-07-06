"""
准备处理人脸数据集，使用Retinaface将人脸检测出裁剪输入arcface中获取embedding
"""
import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os, sys
face_dir = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(face_dir)  # face
sys.path.append(os.path.join(face_dir, 'retinaface'))
sys.path.append(os.path.join(face_dir, 'insight_face'))
# print(sys.path)
import argparse
# face detect 包  调用retinaface
import numpy as np
import cv2
import tqdm
import time
import torch
import torch.backends.cudnn as cudnn
import skimage.transform

# face detect
from core.face.retinaface.data import cfg_mnet, cfg_re50
from core.face.retinaface.layers.functions.prior_box import PriorBox
from core.face.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from core.face.retinaface.utils.box_utils import decode, decode_landm
from core.face.retinaface.models.retinaface import RetinaFace

# face_embedding 包   调用insightface
from PIL import Image
from pathlib import Path
from multiprocessing import Process, Pipe, Value

from core.face.insight_face.config import get_config
from core.face.insight_face.Learner import face_learner
from core.face.insight_face.insight_utils import load_facebank, draw_box_name, prepare_facebank
from core.face.insight_face.insight_model import l2_norm

import base_config
import re

##判断是否有中文
def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    word = word.encode('utf8').decode()
    match = zh_pattern.search(word)
    return match

# 单例模式
def singleton(cls):
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

@singleton
class EmbeddingFace:
    def __init__(self):
        # detect 超参数
        self.trained_model =os.path.join(base_config.root_dir, r'core\weights\retinaface\mobilenet0.25_Final.pth')
        self.network = 'mobile0.25'  # 'resnet50' 'mobile0.25'
        self.cpu = False
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.save_image = False
        self.vis_thres = 0.6
        self.desiredFaceHeight=112
        # embedding 超参数
        # facebank路径 按名称进行命名  提取特征后在该路径保存name和feature
        # 修改为自己需要的目录
        # self.prepare_image_path = os.path.join(base_config.data_rootdir,'extract_face')
        self.prepare_image_path = base_config.extract_face_dir
        self.threshold = 1.54  # ',help='threshold to decide identical faces',default=1.54, type=float)
        self.update = True  # 更新facebank
        self.tta = False  # ", help="whether test time augmentation",action="store_true")
        self.score = False  # ", help="whether show the confidence score",action="store_true")
        self.begin = 0  # ", help="from when to start detection(in seconds)", default=0, type=int)
        self.duration = 0  # ", help="perform detection for how long(in seconds)", default=0, type=int)

        # 配置embedding 模型
        self.conf = get_config(False)
        self.conf.use_mobilfacenet = True  # 是否使用mobilenet模型
        self.learner = face_learner(self.conf, True)
        self.learner.threshold = self.threshold
        if self.conf.device.type == 'cpu':
            print('load cpu model ...')
            # learner.load_state(conf, 'cpu_final.pth', True, True)
            self.learner.load_state(self.conf, 'final_mobile.pth', True, True)
        else:
            print('load gpu model ...')
            self.learner.load_state(self.conf, 'final_mobile.pth', True, True)  # 加载指定路径下 Resnet：final_resnet50.pth
        self.learner.model.eval()
        print('embedding learner loaded')

        torch.set_grad_enabled(False)
        self.cfg = None
        if self.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.network == "resnet50":
            self.cfg = cfg_re50
        self.retina_net_struct = RetinaFace(cfg=self.cfg, phase='test')
        self.retina_net = self.load_model(self.retina_net_struct, self.trained_model, self.cpu)
        self.retina_net.eval()
        #     print('Finished loading model!')
        #     print(net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.cpu else "cuda:0")
        self.retina_net = self.retina_net.to(self.device)

        # Model parameters
        self.image_w = 112
        self.image_h = 112
        self.channel = 3
        self.emb_size = 512

    def load_model(self,model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    # 加载人脸特征
    def load_my_facebank(self,conf=None):
        embeddings = torch.load(os.path.join(self.prepare_image_path, 'facebank.pth'))
        names = np.load(os.path.join(self.prepare_image_path, 'name.npy'))
        return embeddings, names

    # 获取人脸嵌入特征
    def return_facebank(self,update=False):
        if update:
            targets, names = self.prepare_my_facebank(self.conf, self.learner.model, tta=False)
            print('facebank updated')
        else:
            targets, names = self.load_my_facebank(self.conf)
            print('facebank loaded')
        return targets, names

    # 提取人脸特征
    def prepare_my_facebank(self,conf, model, tta=True):
        model.eval()
        embeddings = []
        embedding_numpy = []
        names = ['Unknown']  # 留一个位置给unkn
        for folder_name in tqdm.tqdm(os.listdir(self.prepare_image_path)):
            folder_path = os.path.join(self.prepare_image_path, folder_name)
            print('folder_path',folder_path)
            if os.path.isfile(folder_path):  # 若是文件则跳过  应该按名称命名的文件夹
                continue
            else:
                embs = []
                for image_name in tqdm.tqdm(os.listdir(folder_path)):
                    image_full_path = os.path.join(folder_path, image_name)
                    if not os.path.isfile(image_full_path):
                        continue
                    else:
                        face_image = self.detect_face(image=image_full_path)  # 返回的是GBR 格式图像
                        if face_image is None:
                            continue
                        img = face_image[:, :, [2, 1, 0]]
                        #                     img=np.transpose(face_image,(2,0,1))
                        with torch.no_grad():
                            if tta:
                                mirror = trans.functional.hflip(img)
                                emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                                emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                                embs.append(l2_norm(emb + emb_mirror))
                            else:
                                embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
            if len(embs) == 0:
                continue
            embedding = torch.cat(embs).mean(0, keepdim=True)
            print(type(embedding), embedding.shape)
            # 在文件夹中保存numpy矩阵
            feature_numpy = torch.cat(embs).cpu().numpy()
            print('save numpy mat addr', folder_path)
            np.save(os.path.join(folder_path, 'embedding_feature.npy'), feature_numpy)
            print(type(feature_numpy), feature_numpy.shape)
            embedding_numpy.append(feature_numpy)
            embeddings.append(embedding)
            names.append(folder_name)
        # embeddings = torch.cat(embeddings)
        # names = np.array(names)
        # torch.save(embeddings, os.path.join(self.prepare_image_path, 'facebank.pth'))
        # np.save(os.path.join(prepare_image_path, 'name.npy'), names)
        return embeddings, names

    # 获取边缘框和关键点
    def get_bbox(self,image):
        image = np.float32(image)
        net = self.retina_net
        device = self.device
        resize = 1
        try:
            im_height, im_width, _ = image.shape
        except Exception:
            print('error image', image)
            return None
        scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        image -= (104, 117, 123)
        img = image.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('detect net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets


    def detect_face(self,image=None, show_image=False,save=False, image_path=None,align_method=1):
        """
        检测并对齐人脸
        :param image:图像
        :param
        :param output_image_path:
        :return: 对齐后的人脸
        """
        if save:
            image_dir, image_name = os.path.split(image_path)[0:2]
            if not os.path.exists(os.path.join(image_dir, 'align_face')):
                os.mkdir(os.path.join(image_dir, 'align_face'))
            save_image_path = os.path.join(image_dir, 'align_face', image_name)
        if isinstance(image, str) :
            image = cv2.imread(image)
            # image = np.float32(image)
        else:
            image = image
        try:
            im_height, im_width = image.shape[:2]
        except Exception:
            print('error image,path', image)
            return None
        dets = self.get_bbox(image)
        # show image
        for b in dets:  # b 0~3 为xmin ymin xmax ymax 4 为threshold  后面连续五个-点为地标
            if b[4] < self.vis_thres:
                continue
            # 仅处理一张图片只有一个人脸的情况  因为图片经过剪切
            crop_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2]), :]
            # print('crop_image.shape',crop_image.shape)
            #             resize_crop_image = cv2.resize(crop_image,(112,112))
            # 执行人脸对齐 point 按照 x1 x2 x3 x4 x5 y1 y2 y3 y4 y5
            point = [int(i) for i in
                     [b[5] - b[0], b[7] - b[0], b[9] - b[0], b[11] - b[0], b[13] - b[0], b[6] - b[1], b[8] - b[1],
                      b[10] - b[1], b[12] - b[1], b[14] - b[1]]]
            point2  =np.array([[b[5],b[6]],
                               [b[7],b[8]],
                               [b[9], b[10]],
                               [b[11], b[12]],
                               [b[13],b[14]]])
            #             return crop_image ,point
            facial5points = np.array(point)
            align_obj = FaceAlignerCv2(keypoints=point2)
            # 扩大检测框
            enlarge_bbox,enlarge_image = align_obj.enlarge_crop(b[0:4],ratio=3,image=image)
            # print('enlarge_image shape',enlarge_image.shape)
            enlarget_kp = align_obj.keypoints-[enlarge_bbox[0],enlarge_bbox[1]]
            if align_method==1:
                ########### 一次检测 将检测框一起旋转球旋转后的检测框
                ##在这里返回就可以了 align之后的位置[0:112,0:112,:]就是可用的人脸
                align_enlarge_image,M = align_obj.align(image=enlarge_image, keypoints=enlarget_kp, ratio=1,return_M=True)
                if save:
                    print('save image', save_image_path)
                    if contain_zh(save_image_path):
                        cv2.imencode('.jpg', align_enlarge_image[0:112,0:112,:])[1].tofile(save_image_path)  # 写入中文路径
                    else:
                        cv2.imwrite(save_image_path, align_enlarge_image[0:112,0:112,:])
                if show_image:
                    # 显示 5点用于验证
                    cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
                    cv2.imshow('align_enlarge_image crop', align_enlarge_image[0:112, 0:112, :])
                    cv2.imshow('image', image)
                    cv2.imshow('crop_image', crop_image)
                    cv2.imshow('enlarge_image', enlarge_image)
                    cv2.imshow('align_enlarge_image', align_enlarge_image)

                    cv2.waitKey(0)
                return align_enlarge_image[0:112,0:112,:]

            elif align_method==2:
                ######### 两次检测的方法
                # 对齐
                align_enlarge_image = align_obj.align(image=enlarge_image,keypoints=enlarget_kp,ratio=2)
                #再此检测
                # print('align_enlarge_image shape', align_enlarge_image.shape)
                dets2 = self.get_bbox(image=align_enlarge_image)
                if len(dets2)==0:
                    return None
                else:
                    # 仅处理一张图片只有一个人脸的情况  因为图片经过剪切
                    print('len(dets)2,dets2',len(dets2),dets2)
                    for bb in dets2:  # b 0~3 为xmin ymin xmax ymax 4 为threshold  后面连续五个-点为地标
                        if bb[4] < self.vis_thres:
                            continue
                        print('bb',bb)
                        crop_image2 = align_enlarge_image[max(int(bb[1]),0):int(bb[3]), max(int(bb[0]),0):int(bb[2]), :]
                        print('crop_image2 shape',crop_image2.shape)
                        # 如果crop_image2 过于小则适当放大图片
                        if crop_image2.shape[0] <= 16 or crop_image2.shape[1] <= 16:
                            return None
                        if crop_image2.shape[1]<self.desiredFaceHeight and crop_image2.shape[0]<self.desiredFaceHeight:

                            enlarge_ratio = self.desiredFaceHeight/max(crop_image2.shape[0:2])

                            enlarge_bbox2,enlarge_crop_image2 = align_obj.enlarge_crop(bb,enlarge_ratio,align_enlarge_image)
                            print('enlarge_crop_image2 shape', enlarge_crop_image2.shape)
                            enlarge_crop_image2 = cv2.resize(enlarge_crop_image2,(112,112))
                        else:
                            enlarge_crop_image2 = crop_image2
                            enlarge_crop_image2= cv2.resize(enlarge_crop_image2,(112,112))
                        if save:
                            print('save image',save_image_path)
                            if contain_zh(save_image_path):
                                cv2.imencode('.jpg', enlarge_crop_image2)[1].tofile(save_image_path)  # 写入中文路径
                            else:
                                cv2.imwrite(save_image_path,enlarge_crop_image2)
                        if show_image:
                            # 显示 5点用于验证
                            cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
                            cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
                            cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
                            cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
                            cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
                            cv2.imshow('align_enlarge_image crop', align_enlarge_image[0:112, 0:112, :])
                            cv2.imshow('image', image)
                            cv2.imshow('crop_image', crop_image)
                            cv2.imshow('enlarge_image', enlarge_image)
                            cv2.imshow('align_enlarge_image', align_enlarge_image)
                            cv2.imshow('crop_image2', crop_image2)
                            cv2.imshow('enlarge_crop_image2', enlarge_crop_image2)
                            cv2.waitKey(0)
                        return enlarge_crop_image2

            #             print(crop_image.shape,facial5points)
            resize_crop_image = self.align_face(crop_image, facial5points)


            # text = "{:.4f}".format(b[4])
            # b = list(map(int, b))
            # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # cx = b[0]
            # cy = b[1] + 12
            # cv2.putText(img_raw, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            #
            # # landms
            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

    # 执行人脸对齐
    def align_face(self,img, facial5points):
        facial5points = np.reshape(facial5points, (2, 5))
        crop_size = (self.image_h, self.image_w)
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        output_size = (self.image_h, self.image_w)
        #     print('output_size',output_size)
        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding,
                                                     default_square)
        #     print('reference_5pts',reference_5pts)
        # dst_img = warp_and_crop_face(raw, facial5points)
        dst_img = warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
        #     print('dst_img',dst_img.shape)
        return dst_img

    def embedding_one_image(self,conf, model, tta=True, input_image_path=None):
        face_image = self.detect_face(input_image_path=input_image_path)  # 返回的是GBR 格式图像
        if face_image is None:
            return None
        img = face_image[:, :, [2, 1, 0]]
        #                     img=np.transpose(face_image,(2,0,1))
        with torch.no_grad():
            if tta:
                mirror = trans.functional.hflip(img)
                emb = model(self.conf.test_transform(img).to(self.conf.device).unsqueeze(0))
                emb_mirror = model(self.conf.test_transform(mirror).to(self.conf.device).unsqueeze(0))
                return l2_norm(emb + emb_mirror)
            else:
                return model(self.conf.test_transform(img).to(self.conf.device).unsqueeze(0))

    def get_one_image_embedding(self, tta=False, image=None,image_path=None,save=False):
        conf = self.conf
        model = self.learner.model
        face_image = self.detect_face(image=image,image_path=image_path,save=save)  # 返回的是GBR 格式图像
        if face_image is None:
            return None
        img = face_image[:, :, [2, 1, 0]]
        #                     img=np.transpose(face_image,(2,0,1))
        with torch.no_grad():
            if tta:
                mirror = trans.functional.hflip(img)
                emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                return l2_norm(emb + emb_mirror)
            else:
                return model(conf.test_transform(img).to(conf.device).unsqueeze(0))

    def get_one_image_embedding_direct(self,tta=False, image_path=None):
        conf = self.conf
        model = self.learner.model
        if contain_zh(image_path):
            image_cv2 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        else:
            image_cv2 = cv2.imread(image_path)
        image_cv2 = cv2.resize(image_cv2,(112,112))
        img = image_cv2[:, :, [2, 1, 0]]
        print(img.shape)
        #                     img=np.transpose(face_image,(2,0,1))
        with torch.no_grad():
            if tta:
                mirror = trans.functional.hflip(img)
                emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                return l2_norm(emb + emb_mirror)
            else:
                return model(conf.test_transform(img).to(conf.device).unsqueeze(0))


class FaceAlignerCv2:
    def __init__(self, keypoints, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=112, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.keypoints = np.asarray(keypoints)
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, keypoints, ratio=None,w_h=None,return_M=False):
        # convert the landmark (x, y)-coordinates to a NumPy array

        # 68点对应坐标和5点对应坐标
        if (len(keypoints) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = 36, 42
            (rStart, rEnd) = 42, 48
        else:
            (lStart, lEnd) = 0, 1
            (rStart, rEnd) = 1, 2

        leftEyePts = keypoints[lStart:lEnd]
        rightEyePts = keypoints[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))
        print('angle', angle)
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        if not ratio:
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        else:
            (w, h) = (self.desiredFaceWidth * ratio, self.desiredFaceHeight * ratio)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        if return_M:
            return output,M[:2]
        # return the aligned face
        else:
            return output

    def enlarge_crop(self, bbox, ratio, image):
        """
        :param bbox:[x_min,y_min,x_max,y_max]
        :param ratio: 缩放比例
        :return: 缩放后的
        """
        h, w = image.shape[:2]
        print('image_h_w',h, w)
        enlarge_w = (bbox[2] - bbox[0]) * (ratio - 1)
        enlarge_h = (bbox[3] - bbox[1]) * (ratio - 1)
        enlarge_bbox = [bbox[0] - enlarge_w / 2, bbox[1] - enlarge_h / 2, bbox[2] + enlarge_w / 2,
                        bbox[3] + enlarge_h / 2]

        if enlarge_bbox[0] < 0:
            enlarge_bbox[0] = 0
        else:
            enlarge_bbox[0] = int(enlarge_bbox[0])

        if enlarge_bbox[1] < 0:
            enlarge_bbox[1] = 0
        else:
            enlarge_bbox[1] = int(enlarge_bbox[1])

        if enlarge_bbox[2] > w:
            enlarge_bbox[2] = int(w)
        else:
            enlarge_bbox[2] = int(enlarge_bbox[2])

        if enlarge_bbox[3] > h:
            enlarge_bbox[3] = int(h)
        else:
            enlarge_bbox[3] = int(enlarge_bbox[3])

        print('enlarge_bbox',enlarge_bbox)
        enlarge_image = image[enlarge_bbox[1]:enlarge_bbox[3], enlarge_bbox[0]:enlarge_bbox[2], :]
        return enlarge_bbox,enlarge_image

    def align_face(self):
        # 安装face_alignment
        #测试时使用
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
        image_path = r'C:\PythonProject\find_people\data\test_data\wjc.jpg'
        input_img = io.imread(image_path)
        start_time = time.time()
        preds = fa.get_landmarks(input_img)[-1]
        print('cost time ', time.time() - start_time)
        ratio = 3
        obj = FaceAlignerCv2(preds)
        image_cv2 = cv2.imread(image_path)
        # print(min(preds[:,1]))
        rect = [int(i) for i in [min(preds[:, 0]), min(preds[:, 1]), max(preds[:, 0]), max(preds[:, 1])]]
        crop_image = image_cv2[rect[1]:rect[3], rect[0]:rect[2], :]
        print(rect)
        enlarge_bbox = enlarge_crop(rect, ratio, image_cv2)
        cv2.imshow('crop_image', crop_image)
        enlarge_crop_image = image_cv2[enlarge_bbox[1]:enlarge_bbox[3], enlarge_bbox[0]:enlarge_bbox[2], :]
        cv2.imshow('enlarge_crop_image', enlarge_crop_image)
        keypoints1 = obj.keypoints - np.asarray([int(min(preds[:, 0])), int(min(preds[:, 1]))])
        out_image1 = obj.align(crop_image, keypoints1)
        cv2.imshow('out_image1', out_image1)

        keypoints2 = obj.keypoints - np.asarray([enlarge_bbox[0], enlarge_bbox[1]])
        out_image2 = obj.align(enlarge_crop_image, keypoints2, ratio=ratio)
        cv2.imshow('out_image2', out_image2)
        print(out_image2.shape)
        preds2 = fa.get_landmarks(out_image2[:, :, [2, 0, 1]])[-1]

        rect2 = [int(i) for i in [min(preds2[:, 0]), min(preds2[:, 1]), max(preds2[:, 0]), max(preds2[:, 1])]]
        if rect2[0] < 0:
            rect2[0] = 0
        if rect2[1] < 0:
            rect2[1] = 0
        if rect2[2] > out_image2.shape[1]:
            rect2[2] = int(out_image2.shape[1])
        if rect2[3] > out_image2.shape[0]:
            rect2[3] = int(out_image2.shape[0])

        crop_image3 = out_image2[rect2[1]:rect2[3], rect2[0]:rect2[2], :]
        print('rect2', rect2)
        # print(crop_image3)
        cv2.imshow('out_image3', crop_image3)
        cv2.waitKey(0)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}





# 人脸对齐
# reference facial points, a list of coordinates (x,y)
from skimage import transform as trans

REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    # print('---> default:')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    if (output_size and
            output_size[0] == tmp_crop_size[0] and
            output_size[1] == tmp_crop_size[1]):
        # print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
        return tmp_5pts

    if (inner_padding_factor == 0 and
            outer_padding == (0, 0)):
        if output_size is None:
            print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
            and output_size is None):
        output_size = tmp_crop_size * \
                      (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        print('              deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    # 1) pad the inner region according inner_padding_factor
    # print('---> STEP1: pad the inner region according inner_padding_factor')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 2) resize the padded inner region
    # print('---> STEP2: resize the padded inner region')
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    # print('              crop_size = ', tmp_crop_size)
    # print('              size_bf_outer_pad = ', size_bf_outer_pad)

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding)'
                                '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    # print('              resize scale_factor = ', scale_factor)
    tmp_5pts = tmp_5pts * scale_factor
    #    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
    #    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = size_bf_outer_pad
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    # print('---> STEP3: add outer_padding to make output_size')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)
    #
    # print('===> end get_reference_facial_points\n')

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


def warp_and_crop_face(src_img,  # BGR
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='smilarity'):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(output_size,
                                                        inner_padding_factor,
                                                        outer_padding,
                                                        default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape  # (5,2)
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape  # (5,2)
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type is 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    #        print('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
    elif align_type is 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    #        print('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
    else:
        # tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, ref_pts)
        tfm = tform.params[0:2, :]

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img  # BGR


if __name__ == '__main__':
    obj = EmbeddingFace()
    # obj.return_facebank(update=True)
    # obj.get_one_image_embedding(image=r'D:\dataset\crawler\extract_face\33bc0927-6da8-4fa5-9a82-c6735debe4c0\1_1.0.jpg')
    # temp_dir = r'D:\dataset\crawler\extract_face\9ec1bcd3-fa58-4546-a4dd-f8cb381ec643'
    # for d in [os.path.join(temp_dir,image_name) for image_name in os.listdir(temp_dir)]:
    #     obj.detect_face(image=d,show_image=True)
    # cv2.destroyAllWindows()
    # r'D:\dataset\crawler\facebank\黄米依\
    # obj.detect_face(image=r'D:\dataset\crawler\facebank\15_1.0.jpg',show_image=True)
    # obj.detect_face(image=r'D:\dataset\crawler\extract_face\33bc0927-6da8-4fa5-9a82-c6735debe4c0\1_1.0.jpg',show_image=True)
    # obj.detect_face(image=r'D:\dataset\crawler\1.png',show_image=True)