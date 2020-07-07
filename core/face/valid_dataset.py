"""
在数据集上评估模型
"""
import os
import numpy as np
import tqdm
import re
import random
import cv2
import glob
import json

from core.face.embedding_face import EmbeddingFace
# 评估lfw数据集 路径
pairs_path = r'D:\dataset\face\lfw\pairs.txt'
lfw_dir = r'D:\dataset\face\lfw\lfw-112X96'


# def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
#     # Calculate evaluation metrics
#     thresholds = np.arange(0, 4, 0.01)
#     embeddings1 = embeddings[0::2]
#     embeddings2 = embeddings[1::2]
#     tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
#                                                np.asarray(actual_issame), nrof_folds=nrof_folds,
#                                                distance_metric=distance_metric, subtract_mean=subtract_mean)
#     thresholds = np.arange(0, 4, 0.001)
#     val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
#                                               np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
#                                               distance_metric=distance_metric, subtract_mean=subtract_mean)
#     return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)  # 是否为同一人
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def valid_lfw():
    pairs = read_pairs(pairs_path)
    print(pairs)
    path_list, issame_list = get_paths(lfw_dir=lfw_dir,pairs=pairs)
    print(len(path_list),type(path_list[0]))
    print(len(issame_list))
    obj = EmbeddingFace()
    error_pairs=0
    right_pairs=0
    true_distance=[]
    false_distance=[]
    for i in tqdm.tqdm([j for j in range(len(path_list))][::2]):
        pair0_path = path_list[i]
        pair1_path = path_list[i + 1]
        is_same = issame_list[int(i/2)]
        print('pair0_path',pair0_path)
        print('pair1_path',pair1_path)
        # 使用对齐人脸 准确率97.73%
        # pair0_feature = obj.get_one_image_embedding_direct(input_image_path=pair0_path)
        # pair1_feature = obj.get_one_image_embedding_direct(input_image_path=pair1_path)

        # 使用对齐人脸先检测在提取特征 准确率87.91%
        pair0_feature = obj.get_one_image_embedding(image=pair0_path)
        pair1_feature = obj.get_one_image_embedding(image=pair1_path)
        if pair0_feature is not None and pair1_feature is not None:
            pair0_feature = pair0_feature.cpu().numpy()
            pair1_feature = pair1_feature.cpu().numpy()
            distance = np.sum(np.square(pair0_feature - pair1_feature))
            print(distance)
            if is_same:
                true_distance.append(distance)
            else:
                false_distance.append(distance)
            if distance<1.35:
                if is_same:
                    right_pairs+=1
            else:
                if not is_same:
                    right_pairs += 1
        else:
            error_pairs+=1
    print('true_distance',sum(true_distance)/len(true_distance))
    print('flase_distance',sum(false_distance)/len(false_distance))
    print('accuracy:',right_pairs,right_pairs/6000)
    print('error_pairs:',error_pairs)

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

def valid_self_data():
    obj = EmbeddingFace()

    root_data_dir = r'D:\dataset\crawler\facebank'
    text_images = {}
    train_images = {}
    images_feature = {}
    for dir_name in tqdm.tqdm(os.listdir(root_data_dir)):
        sub_dir = os.path.join(root_data_dir,dir_name)
        image_name_list = glob.glob(os.path.join(sub_dir,'*.jpg'))

        feature = []
        for i,image_name in enumerate(image_name_list):
            # if i< len(image_name_list)*split_ratio:
            image_path = os.path.join(sub_dir,image_name)
            if contain_zh(image_path):
                image_cv2 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            else:
                image_cv2 = cv2.imread(image_path)
            pair0_feature = obj.get_one_image_embedding(image=image_cv2,image_path=image_path,save=True)
            if pair0_feature is None:
                pair0_feature = [-1]*512
            else:
                pair0_feature = pair0_feature.cpu().numpy().tolist()
            images_feature.update({dir_name+'_'+image_name:pair0_feature})
        with open('facebank.json','w') as f:
            json.dump(images_feature,f)

def embedding_self_data(split_ratio=0.2):
    obj = EmbeddingFace()
    root_data_dir = r'D:\dataset\crawler\facebank'
    for dir_name in tqdm.tqdm(os.listdir(root_data_dir)):
        sub_dir = os.path.join(root_data_dir,dir_name)
        if os.path.isdir(sub_dir):
            image_name_list = glob.glob(os.path.join(sub_dir,'*.jpg'))
            feature = []
            for i,image_name in enumerate(image_name_list):
                image_path = os.path.join(sub_dir,image_name)
                pair0_feature = obj.get_one_image_embedding_direct(image_path=image_path)
                pair0_feature = pair0_feature.cpu().numpy()
                feature.append(pair0_feature)
            print('save path',os.path.join(sub_dir,'align_face_feature.npy'))
            np.save(os.path.join(sub_dir,'align_face_feature.npy'),np.asarray(feature))

map_dict={'黄米依':'hmy','张颂文':'zsw','李梦':'lm','刘琳':'ll','芦芳生':'lsf','秦昊':'qh','荣梓杉':'rzs','史彭元':'spy','王景春':'wjc','王圣迪':'wsd'}
def valid_video(video_path):
    cap = cv2.VideoCapture(video_path)
    obj =EmbeddingFace()

    while True:
        flag,frame = cap.read()
        if not flag:
            print('video is over')
            break
        search_image_feature=obj.get_one_image_embedding(image=frame)
        if search_image_feature is None:
            cv2.imshow('frmae', frame)
            cv2.waitKey(1)
        else:
            search_image_feature = search_image_feature.cpu().numpy()
            # 提取人脸文件名
            name_list = os.listdir(r'D:\dataset\crawler\facebank')
            hit_count = [0] * len(name_list)
            for index, dir_name in enumerate(name_list):
                save_feature_path = os.path.join(r'D:\dataset\crawler\facebank', dir_name, 'align_face_feature.npy')
                features = np.load(save_feature_path)
                print('dir_name, features.shape',dir_name, features.shape)
                for i in range(features.shape[0]):
                    base_features = np.squeeze(features[i,:,:])
                    cos_distance =cos_sim(base_features,search_image_feature)
                    # print('cos_distance',cos_distance)
                    # 若距离小于0.8判断为相识
                    if cos_distance > 0.746:
                        hit_count[index] += 1
            print('hit_count',hit_count)
            if max(hit_count)<2:
                continue
            max_index = hit_count.index(max(hit_count))
            print('####################name_list[max_index]',name_list[max_index])
            # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,map_dict[name_list[max_index]], (100, 100), font, 5, (200, 255, 155), 2, cv2.LINE_AA)
            cv2.imshow('frmae',frame)
            cv2.waitKey(1)


#  直接测试
def te():
    right_pairs = 0
    all_pairs=0
    with open('facebank.json', 'r') as f:
        images_feature = json.load(f)
    pre_name,pre_frature= None,None
    cur_name,cur_frature= None,None
    for name, pair0_feature in images_feature.items():
        if pair0_feature.count(-1)==512:
            continue
        name =  name.split('_')[0]
        pair0_feature = np.array(pair0_feature)
        if pre_frature is None and cur_frature is None:
            cur_name = name
            cur_frature = pair0_feature
            continue
        else:
            pre_name = cur_name
            pre_frature = cur_frature
            cur_name = name
            cur_frature = pair0_feature
        if pre_name==name:
            is_same=True
        else:
            is_same=False
        print('name,prename', name,pre_name)
        distance = np.sum(np.square(cur_frature - pre_frature))
        all_pairs+=1
        print('distance',distance)
        if is_same:
            if distance < 1.35:
                right_pairs += 1

        else:
            if distance < 1.35:
                right_pairs += 1

    print('accuracy',right_pairs/all_pairs*100,right_pairs,all_pairs)

# 先生成lfw类似pairs，使用json保存
def generate_self_datasets_pairs(save_path=None,pairs=(1000,1000)):
    data_root_dir = r'D:\dataset\crawler\facebank'
    # 存储所有的图像对
    pair_list=[]
    # 存储所有的图像路径 生成不同人名下的对
    all_image_list = []
    for sub_name in tqdm.tqdm(os.listdir(data_root_dir)):
        sub_dir = os.path.join(data_root_dir,sub_name,'align_face')
        if not os.path.exists(sub_dir):
            print('sub_dir',sub_dir)
            raise Exception('对齐人脸路径不存在')

        image_list = os.listdir(sub_dir)
        image_counts = len(image_list)
        assert image_counts>1,'对齐人脸文件夹中不存在图片'
        # 生存累内的对
        all_image_list.append([os.path.join(sub_dir,image_name) for image_name in image_list])
        if image_counts==1:
            # 只有1个
            pass
        else:
            for i in range(image_counts):
                # 跳过一些图片
                if i%2!=0:
                    continue
                for j in range(1,image_counts):
                    # 跳过一些图片
                    if j % 2 == 0:
                        continue
                    pair0 = os.path.join(sub_dir,image_list[i])
                    pair1 = os.path.join(sub_dir,image_list[j])
                    pair_list.append([pair0,pair1,True])

    # 生成类外的对
    for i in range(len(all_image_list)):
        for j in range(1,len(all_image_list)):
            if i!=j:
                min_len = min(len(all_image_list[i]),len(all_image_list[j]))
                for k in range(min_len):
                    pair0 = all_image_list[i][k]
                    pair1 = all_image_list[j][k]
                    pair_list.append([pair0, pair1, False])
    print('len pair_list',len(pair_list))
    with open('pairs.json','w') as f:
        json.dump(pair_list,f)

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

## acc 61% hard
def valid_pairs():
    with open('pairs.json','r')as f:
        pairs = json.load(f)
    # 存储真和假距离
    true_distance = []
    false_distance = []
    true_cos_distance=[]
    false_cos_distance=[]
    right_pairs=0
    error_pairs=0
    for pair in tqdm.tqdm(pairs):
        pair0_path = pair[0]
        pair1_path = pair[1]
        is_same = pair[2]
        print('pair0_path',pair0_path)
        print('pair1_path',pair1_path)
        # 使用对齐人脸 准确率97.73%
        obj = EmbeddingFace()
        pair0_feature = obj.get_one_image_embedding_direct(image_path=pair0_path)
        pair1_feature = obj.get_one_image_embedding_direct(image_path=pair1_path)

        # 使用对齐人脸先检测在提取特征 准确率87.91%
        # pair0_feature = obj.get_one_image_embedding(image=pair0_path)
        # pair1_feature = obj.get_one_image_embedding(image=pair1_path)
        if pair0_feature is not None and pair1_feature is not None:
            pair0_feature = pair0_feature.cpu().numpy()
            pair1_feature = pair1_feature.cpu().numpy()
            distance = np.sum(np.square(pair0_feature - pair1_feature))
            cos_distance =cos_sim(pair0_feature,pair1_feature)
            print('cos_distance',cos_distance)
            print(distance)
            if is_same:
                true_distance.append(distance)
                true_cos_distance.append(cos_distance)
            else:
                false_distance.append(distance)
                false_cos_distance.append(cos_distance)
            if cos_distance>0.746:
                if is_same:
                    right_pairs+=1
            else:
                if not is_same:
                    right_pairs += 1
        else:
            error_pairs+=1
    print('true_distance',sum(true_distance)/len(true_distance))
    print('flase_distance',sum(false_distance)/len(false_distance))
    print('accuracy:',right_pairs,right_pairs/len(pairs),right_pairs,right_pairs,len(pairs))
    print('error_pairs:',error_pairs)

    print('true_cos_distance',sum(true_cos_distance)/len(true_cos_distance))
    print('flase_cos_distance',sum(false_cos_distance)/len(false_cos_distance))



if __name__=='__main__':
    # valid_lfw()
    #  对齐人脸
    # valid_self_data()
    # 人脸嵌入
    embedding_self_data()
    # 生成验证对
    # generate_self_datasets_pairs()
    # 验证准确率
    valid_pairs()

    # 视频上测试
    # valid_video(video_path=r'D:\dataset\crawler\ym_jiedu.flv')
    # np_data = np.load(r'D:\dataset\crawler\facebank\黄米依\align_face_feature.npy',allow_pickle=True)
    # print(np_data.shape)















