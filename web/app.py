import os, sys

# 主动添加路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'core', 'face'))
sys.path.append(os.path.join(root_dir, 'core', 'face', 'retinaface'))
sys.path.append(os.path.join(root_dir, 'core', 'face', 'insight_face'))
# print(sys.path)
import uuid
import json
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import time
import random
import sys
import os, urllib, cv2
from PIL import Image

from core.face import embedding_face
import base_config

# 人脸匹配阈值
distance_threshold = 1.0
# 文件夹命中阈值
count_thresold = 3


@st.cache
def load_model(h=0):
    print(h)
    obj = embedding_face.EmbeddingFace()
    return obj


def main():
    #
    # 当要显示图片过多是选择分页
    add_selectbox = st.sidebar.selectbox(
        "选择分页",
        ("0", "1", "2")
    )
    # 设置显示图片数量
    show_image_number = st.sidebar.slider('show image number', 1, 100, value=20)
    # 检测
    detect_flag = st.sidebar.checkbox('开始检测', value=False)

    image_file = st.file_uploader("上传图片", type=['jpg', 'png'])
    video_file = st.text_input('Enter a file path:')
    if video_file is not None:
        cap = cv2.VideoCapture(video_file)
    # print(uploaded_file)
    # uploaded_file = r'D:\dataset\crawler\extract_face\a7a3e47c-4040-4c85-a2cc-775476de673b\400_1.0.jpg'
    if image_file is not None and detect_flag:
        print('upload_file:', image_file)
        image = Image.open(image_file)
        image.save('temp.jpg')
        st.image(image, caption='上传图片', width=200)
        wait_text = st.subheader('search image...')
        embedding_face = load_model(h=0)
        search_image_feature = embedding_face.get_one_image_embedding(
            input_image_path='temp.jpg')
        if search_image_feature is None:
            st.header('检测不到人脸')
            return
        # print(search_image_feature)
        search_image_feature = search_image_feature.cpu().numpy()
        # 提取人脸文件名
        name_list = os.listdir(base_config.extract_face_dir)
        hit_count = [0] * len(name_list)
        for index, dir_name in enumerate(name_list):
            print(dir_name)
            save_feature_path = os.path.join(base_config.extract_face_dir, dir_name, 'embedding_feature.npy')
            if not os.path.exists(save_feature_path):
                continue
            features = np.load(save_feature_path)
            print(dir_name, features.shape)
            for i in range(features.shape[0]):
                distance = np.sqrt(np.sum(np.square(features[i, :] - search_image_feature)))
                # 若距离小于0.8判断为相识
                if distance < distance_threshold:
                    hit_count[index] += 1
        print(hit_count)
        name = []
        uuid_code_list = []
        for i, value in enumerate(hit_count):
            if value >= count_thresold:
                print(name_list[i])
                name.append(name_list[i])
                uuid_code_list.append(name_list[i])
        with open(r'D:\PythonProject\find_people\data\uuid_videos_path.json', 'r') as f:
            uuid_videos_path = json.load(f)
        st.subheader(name)

        # 显示图片
        if image:
            for code in uuid_code_list:
                image_path = os.path.join(base_config.extract_face_dir, code,
                                          os.listdir(os.path.join(base_config.extract_face_dir, code))[0])
                show_image = Image.open(image_file)
                st.image([show_image], caption=['test image1'], width=150)
                show_video = st.checkbox('显示视频', value=False)
                if show_video:
                    print('uuid_code_list', len(uuid_code_list))
                    print('video path', uuid_videos_path[code])
                    st.video(uuid_videos_path[code])
                    st.write(show_video)

    # 测试用不检测 直接显示
    elif image_file is not None and detect_flag is False:
        print('upload_file:', image_file)
        image = Image.open(image_file)
        image.save('temp.jpg')
        st.header('upload')
        st.image(image, caption='上传图片', width=200)
        wait_text = st.subheader('search image...')
        # TODO detect
        time.sleep(2)
        wait_text.empty()
        st.subheader('搜索结果...')
        for i in range(len(show_image_number)):
            st.image(image, caption='confidence:' + str(random.random()), width=200)
            show_video = st.checkbox('show', value=False)

            if show_video:
                video = st.video('C:\PythonProject\dataset\ch06_20191103080800_out_pt.avi')
            if not show_video:
                try:
                    if video:
                        video.empty()
                except Exception:
                    pass


def demo():
    #
    # 选择相似阈值
    option = st.sidebar.selectbox('选择检测视频还是图片', ('image', 'video'))
    threshold = st.sidebar.slider('阈值', 0.0, 2.0, value=0.8)
    # 设置显示图片数量
    show_image_number = st.sidebar.slider('show image number', 1, 100, value=20)
    image_file = st.sidebar.file_uploader("上传图片", type=['jpg', 'png'])
    video_file = st.sidebar.text_input('Enter a file path:')
    # 检测
    detect_flag = st.sidebar.checkbox('开始检测', value=False)
    if option == 'image' and image_file:
        image = Image.open(image_file)
        save_name = 'temp.jpg'
        image.save(save_name)
        st.header('upload')
        st.image(image, caption='上传图片', width=500)
        detect_image(image=save_name, detect_flag=detect_flag, show_image_number=show_image_number)
    elif option == 'video' and video_file:
        detect_video(video_path=video_file, detect_flag=detect_flag, show_image_number=show_image_number)


def detect_image(image, detect_flag, show_image_number):
    if isinstance(image, str):
        image_cv2 = cv2.imread(image)
    elif type(image) is np.ndarray:
        image_cv2 = image
    else:
        raise Exception('image 格式不对')
    wait_text = st.subheader('search image...')
    # TODO detect
    obj = load_model(h=0)
    search_image_feature = obj.get_one_image_embedding(image=image_cv2)
    if search_image_feature is None:
        st.header('检测不到人脸')
        return
    wait_text.empty()
    st.subheader('搜索结果...')
    search_image_feature = search_image_feature.cpu().numpy()

    ## 遍历提取人脸目录
    md5_list = os.listdir(base_config.align_face_dir)
    hit_count = [0] * len(md5_list)
    for index, dir_name in enumerate(md5_list):
        # print(dir_name)
        save_feature_path = os.path.join(base_config.align_face_dir, dir_name, 'embedding_feature.npy')
        if not os.path.exists(save_feature_path):
            continue
        features = np.load(save_feature_path)
        # print(dir_name, features.shape)
        for i in range(features.shape[0]):
            distance = np.sqrt(np.sum(np.square(features[i, :] - search_image_feature)))
            # 若距离小于0.8判断为相识
            if distance < distance_threshold:
                hit_count[index] += 1
    print(hit_count)
    ##排序后选取最大的前10验证
    hit_count = list(np.argsort(hit_count))
    show_hit_count = hit_count[-1:-10:-1]
    md5_list_show = []
    for i in show_hit_count:
        if hit_count[i] >= count_thresold:
            md5_list_show.append(md5_list[i])
    with open(base_config.videos_path_md5_jsonpath, 'r') as f:
        videos_path_md5 = json.load(f)
    ##  显示图片 每个文件夹显示一张
    i=0
    show_videos_flag = []
    show_videos_empty = [st.empty()]*len(md5_list_show)
    for video_path,(video_md5,_,__) in videos_path_md5.items():
        if video_md5 in md5_list_show:
            if i>=50:
                break
            image_path = os.path.join(base_config.extract_face_dir, video_md5,
                                      os.listdir(os.path.join(base_config.extract_face_dir, video_md5))[0])
            show_image = Image.open(image_path)
            st.image([show_image], caption=['test image1'], width=150)
            show_videos_flag.append(st.checkbox('显示视频', value=False,key=i))
            if show_videos_flag[i]:
                print(os.path.join(base_config.row_dir,video_path),show_videos_flag[i])
                # show_videos_empty[i].video(os.path.join(base_config.row_dir,video_path.strip('\\')))
            else:
                show_videos_empty[i].empty()
                pass
            i+=1
    ## 显示图片和显示视频确认框
    # show_videos_list = []
    # show_videos_empty = []
    # for i in range(show_image_number):
    #     st.image(image, caption='conf:' + str(round(random.random(), 2)), width=500)
    #     show_video = st.checkbox('show', value=False, key=uuid.uuid4())
    #     show_videos_list.append(show_video)
    #     show_videos_empty.append(st.empty())
    # for i, show_video in enumerate(show_videos_list):
    #     if show_video:
    #         # show_videos_empty[i].video('C:\PythonProject\dataset\ch06_20191103080800_out_pt.avi')
    #         # show_videos_empty[i].video(r'C:\ChromeDownload\b_test.flv')
    #         show_videos_empty[i].video(r'C:\ChromeDownload\videoplayback.mp4')
    #     else:
    #         show_videos_empty[i].empty()


def detect_video(video_path, detect_flag, show_image_number):
    st.subheader('upload')
    st.video(video_path)
    cap = cv2.VideoCapture(video_path)
    show_frame = []
    frame_count = 0
    while True:
        flag, frame = cap.read()
        if not flag:
            print('video is over')
        if frame_count % 30 == 0:
            show_frame.append(frame)
        if frame_count > 200:
            break
        frame_count += 1
    st.subheader('提取帧：')
    st.image(show_frame, caption=['a'] * len(show_frame), width=200)
    # TODO 将帧送到检测图片中寻找相似
    for frame in show_frame:
        detect_image(frame, detect_flag, show_image_number)


def show_video():
    st.video(r'C:\ChromeDownload\videoplayback.mp4')


if __name__ == "__main__":
    # main()
    demo()
    # show_video()


