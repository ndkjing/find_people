import os, sys
# 主动添加路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'core', 'face'))
sys.path.append(os.path.join(root_dir, 'core', 'face', 'retinaface'))
sys.path.append(os.path.join(root_dir, 'core', 'face', 'insight_face'))
# print(sys.path)

import json
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import time
import sys
import os, urllib, cv2
from PIL import Image

import base_config

# 人脸匹配阈值
distance_threshold = 1.0
# 文件夹命中阈值
count_thresold=2
from core.face import embedding_face_base as embedding_face

@st.cache
def load_model(h=0):
    print(h)
    from core.face import embedding_face_base as embedding_face
    return embedding_face


def main():
    # embedding_face = load_model(h=0)
    # 当要显示图片过多是选择分页
    add_selectbox = st.sidebar.selectbox(
        "选择分页",
        ("0", "1", "2")
    )
    image_file = st.file_uploader("上传图片", type=['jpg', 'png'])
    video_file =  st.text_input('Enter a file path:')
    if video_file is not  None:
        cap = cv2.VideoCapture(video_file)
    # print(uploaded_file)
    # uploaded_file = r'D:\dataset\crawler\extract_face\a7a3e47c-4040-4c85-a2cc-775476de673b\400_1.0.jpg'
    if image_file is not None:
        print('upload_file:', image_file)
        image = Image.open(image_file)
        image.save('temp.jpg')
        st.image(image, caption='上传图片', width=200)

        search_image_feature = embedding_face.get_one_image_embedding(
            input_image_path='temp.jpg')
        if search_image_feature is None:
            st.header('搜索无对应结果')
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
            print(dir_name,features.shape)
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
    if image_file is not None:
        wait_text = st.text('search image...')
    # 设置显示图片数量
    show_image_number = st.slider('show image number', 1, 100, value=20)

    # TODO
    search_finished = False

    # 显示图片

    if image:
        for code in uuid_code_list:
            image_path = os.path.join(base_config.extract_face_dir, code,
                                      os.listdir(os.path.join(base_config.extract_face_dir, code))[0])
            show_image = Image.open(image_file)
            st.image([show_image], caption=['test image1'], width=150)
            show_video = st.checkbox('显示视频',value=False)
            if show_video:
                print('uuid_code_list', len(uuid_code_list))
                print('video path', uuid_videos_path[code])
                st.video(uuid_videos_path[code])
                st.write(show_video)


# def choose_video():
#     # 当要显示图片过多是选择分页
#     add_selectbox = st.sidebar.selectbox(
#         "选择分页",
#         ("0", "1", "2")
#     )
#     uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png'])
#     # print(uploaded_file)
#     # uploaded_file = r'D:\dataset\crawler\extract_face\a7a3e47c-4040-4c85-a2cc-775476de673b\400_1.0.jpg'
#     if uploaded_file is not None:
#         print('upload_file:', uploaded_file)
#         image = Image.open(uploaded_file)
#         image.save('temp.jpg')
#         st.image(image, caption='上传图片', width=200)
#         embedding_face = load_model()
#         search_image_feature = embedding_face.get_one_image_embedding(
#             input_image_path='temp.jpg')
#         if search_image_feature is None:
#             st.header('搜索无对应结果')
#             return
#         # print(search_image_feature)
#         search_image_feature = search_image_feature.cpu().numpy()
#         # 提取人脸文件名
#         name_list = os.listdir(base_config.extract_face_dir)
#         hit_count = [0] * len(name_list)
#         for index, dir_name in enumerate(name_list):
#             print(dir_name)
#             distance_list = [0] * 15
#             save_feature_path = os.path.join(base_config.extract_face_dir, dir_name, 'embedding_feature.npy')
#             if not os.path.exists(save_feature_path):
#                 continue
#             features = np.load(save_feature_path)
#             for i in range(features.shape[0]):
#                 distance = np.sqrt(np.sum(np.square(features[i, :] - search_image_feature)))
#                 # 若距离小于0.8判断为相识
#                 if distance < distance_threshold:
#                     hit_count[index] += 1
#         print(hit_count)
#         uuid_code_list = []
#         for i, value in enumerate(hit_count):
#             if value >= count_thresold:
#                 print(name_list[i])
#                 uuid_code_list.append(name_list[i])
#         with open(r'D:\PythonProject\find_people\data\uuid_videos_path.json', 'r') as f:
#             uuid_videos_path = json.load(f)
#
#     if uploaded_file is not None:
#         wait_text = st.text('search image...')
#     # 设置显示图片数量
#     show_image_number = st.slider('show image number', 1, 100, value=20)
#
#     # TODO
#     search_finished = False
#
#     # 显示图片
#
#     if image:
#         for code in uuid_code_list:
#             image_path = os.path.join(base_config.extract_face_dir, code,
#                                       os.listdir(os.path.join(base_config.extract_face_dir, code))[0])
#             show_image = Image.open(uploaded_file)
#             st.image([show_image], caption=['test image1'], width=150)
#             show_video = st.checkbox('显示视频',value=False)
#             if show_video:
#                 print('uuid_code_list', len(uuid_code_list))
#                 print('video path', uuid_videos_path[code])
#                 st.video(uuid_videos_path[code])
#                 st.write(show_video)

if __name__ == "__main__":
    main()
