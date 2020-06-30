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
print(sys.path)
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
        self.trained_model =os.path.join(base_config.root_dir, 'core/weights/mobilenet0.25_Final.pth')
        self.network = 'mobile0.25'  # 'resnet50'
        self.cpu = False
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.save_image = False
        self.vis_thres = 0.6

        # embedding 超参数
        # facebank路径 按名称进行命名  提取特征后在该路径保存name和feature
        # 修改为自己需要的目录
        # self.prepare_image_path = os.path.join(base_config.data_rootdir,'extract_face')
        self.prepare_image_path = r'C:\PythonProject\find_people\data\facebank'
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
        cfg = None
        if self.network == "mobile0.25":
            cfg = cfg_mnet
        elif self.network == "resnet50":
            cfg = cfg_re50
        self.retina_net_struct = RetinaFace(cfg=cfg, phase='test')
        self.retina_net = self.load_model(self.retina_net_struct, self.trained_model, self.cpu)
        self.retina_net.eval()
        #     print('Finished loading model!')
        #     print(net)
        cudnn.benchmark = True
        device = torch.device("cpu" if self.cpu else "cuda:0")
        self.retina_net = self.retina_net.to(device)

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
            print(folder_path)
            if os.path.isfile(folder_path):  # 若是文件则跳过  应该按名称命名的文件夹
                continue
            else:
                embs = []
                for image_name in tqdm.tqdm(os.listdir(folder_path)):
                    image_full_path = os.path.join(folder_path, image_name)
                    if not os.path.isfile(image_full_path):
                        continue
                    else:
                        face_image = self.detect_face(input_image_path=image_full_path)  # 返回的是GBR 格式图像
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

    def detect_face(self,img=None, input_image_path="./curve/test.jpg", output_image_path='test.jpg'):
        """
        img 输入GBR格式图片矩阵
        input_image_path 输入图片路径
        output_image_path 保存图片路径
        """
        net = self.retina_net
        device = self.device
        resize = 1
        if isinstance(input_image_path, str) and img is None and input_image_path.endswith('jpg'):
            img_raw = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)
        else:
            img_raw = img
        try:
            im_height, im_width, _ = img.shape
        except Exception:
            print('error image', input_image_path)
            return None
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

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

        # show image
        for b in dets:  # b 0~3 为xmin ymin xmax ymax 4 为threshold  后面连续五个点为地标
            if b[4] < vis_thres:
                continue
            # 仅处理一张图片只有一个人脸的情况
            try:
                crop_image = img_raw[int(b[1]):int(b[3]), int(b[0]):int(b[2]), :]

                #             resize_crop_image = cv2.resize(crop_image,(112,112))
                # 执行人脸对齐 point 按照 x1 x2 x3 x4 x5 y1 y2 y3 y4 y5
                point = [int(i) for i in
                         [b[5] - b[0], b[7] - b[0], b[9] - b[0], b[11] - b[0], b[13] - b[0], b[6] - b[1], b[8] - b[1],
                          b[10] - b[1], b[12] - b[1], b[14] - b[1]]]
                #             point  =[int(i) for i in [b[5]-b[0], b[6]-b[1],b[7]-b[0], b[8]-b[1], b[9]-b[0],b[10]-b[1], b[11]-b[0], b[12]-b[1], b[13]-b[0],b[14]-b[1]]]
                #             return crop_image ,point
                facial5points = np.array(point)
                #             print(crop_image.shape,facial5points)
                resize_crop_image = align_face(crop_image, facial5points)
                return resize_crop_image
            except:
                print('exception')
                return None
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

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

    def get_one_image_embedding(self, tta=False, input_image_path=None):
        conf = self.conf
        model = self.learner.model
        face_image = self.detect_face(input_image_path=input_image_path)  # 返回的是GBR 格式图像
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
    obj.return_facebank(update=True)


