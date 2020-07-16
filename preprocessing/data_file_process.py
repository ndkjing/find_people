import os
import sys
import uuid
import base_config
import glob
import json
import time
from tqdm import tqdm
import hashlib
"""
统计文件夹路径和图片信息
"""



def get_md5_code_str(src):
    m1 = hashlib.md5()
    m1.update(src.encode('utf8'))
    return m1.hexdigest()

# 生成图片对应索引
def generate_images_path_map(reload_all=False):
    if reload_all:
        try:
            os.remove(base_config.md5_images_path_jsonpath)
            os.remove(base_config.images_path_md5_jsonpath)
        except Exception:
            pass
    image_md5={}
    md5_images ={}
    images_dir = os.path.join(base_config.row_dir,'images')
    for sub_file in os.listdir(images_dir):
        sub_dir = os.path.join(images_dir,sub_file)
        images_files = glob.glob(os.path.join(sub_dir, "*.png")) + \
                       glob.glob(os.path.join(sub_dir, "*.jpg")) + \
                       glob.glob(os.path.join(sub_dir, "*.webp")) + \
                       glob.glob(os.path.join(sub_dir, "*.bmp"))
        if len(images_files)==0:
            print('文件夹中包含子文件夹')
            for sub1_file in os.listdir(sub_dir):
                sub1_dir = os.path.join(sub_dir, sub1_file)
                images_files = glob.glob(os.path.join(sub1_dir, "*.png")) + \
                               glob.glob(os.path.join(sub1_dir, "*.jpg")) + \
                               glob.glob(os.path.join(sub1_dir, "*.webp")) + \
                               glob.glob(os.path.join(sub1_dir, "*.bmp"))
                if len(images_files) == 0:
                    print(sub1_dir)
                    raise Exception('文件内不存在任何文件')
                md5_code =  make_file_id(str(sub1_dir))
                print(md5_code)
                item = {sub1_dir:[md5_code,False,sub1_file]}
                inverse_item = {md5_code:sub1_dir}
                image_md5.update(item)
                md5_images.update(inverse_item)
            with open(base_config.images_path_md5_jsonpath,'w') as f0:
                json.dump(image_md5,f0)
            with open(base_config.md5_images_path_jsonpath,'w') as f1:
                json.dump(md5_images,f1)


# 生成视频对应索引
def generate_videos_path_map(reload_all=False):
    """
    生成视频数据地址对应uuid_code 是否包含图片  是否已提取人脸
    video_path:[uuid_code,have_images,extracted]
    :param updata:
    :return:
    """
    video_part_path_md5 = os.listdir(os.path.join(base_config.data_rootdir,'extract_face'))
    if reload_all:
        try:
            os.remove(base_config.md5_videos_path_jsonpath)
            os.remove(base_config.videos_path_md5_jsonpath)
        except Exception:
            pass
    try:
        with open(base_config.md5_videos_path_jsonpath,'r') as f0:
            videos_path_md5_json = json.load(f0)
        with open(base_config.videos_path_md5_jsonpath,'r') as f1:
            md5_videos_path_json = json.load(f1)
    except Exception:
        videos_path_md5_json,md5_videos_path_json = {},{}
    assert 'videos' in os.listdir(base_config.row_dir), 'config 中文件夹内不包含videos 文件夹'

    videos_dir = os.path.join(base_config.row_dir, 'videos')

    videos_dirname_lists = os.listdir(videos_dir)

    for videos_dirname in tqdm(videos_dirname_lists):
        # 循环videos文件夹
        videos_sub1_dir = os.path.join(videos_dir, videos_dirname)
        if not os.path.isdir(videos_sub1_dir):  # 若是目录
            print('不在文件夹中文件',videos_sub1_dir)
        images_files = glob.glob(os.path.join(videos_sub1_dir, "*.png")) + \
                       glob.glob(os.path.join(videos_sub1_dir, "*.jpg")) + \
                       glob.glob(os.path.join(videos_sub1_dir, "*.")) + \
                       glob.glob(os.path.join(videos_sub1_dir, "*.bmp"))
        videos_files = glob.glob(os.path.join(videos_sub1_dir, "*.mp4")) + \
                       glob.glob(os.path.join(videos_sub1_dir, "*.avi")) + \
                       glob.glob(os.path.join(videos_sub1_dir, "*.mpg")) + \
                       glob.glob(os.path.join(videos_sub1_dir, "*.webm")) + \
                       glob.glob(os.path.join(videos_sub1_dir, "*.mpeg"))

        if len(videos_files) ==0:
            print('文件夹下没有视频',videos_sub1_dir)
        if len(images_files)!=0:
            print(videos_sub1_dir,'是包含图片的文件夹')
            for video_name in videos_files:
                video_path = os.path.join(videos_sub1_dir,video_name)
                video_part_path = video_path.split('row')[1]
                print('video_path', video_path)
                if video_part_path in videos_path_md5_json.keys():
                    print('已包含该视频',video_path)
                else:
                    md5_code =get_md5_code_str(video_part_path)
                    # 视频路径：[md5code 有图片  已提取]
                    md5_code =get_md5_code_str(video_part_path)
                    extracted = True if md5_code in video_part_path_md5 else False
                    # 视频路径：[md5code 有图片  已提取]
                    videos_path_md5_json[video_part_path]=[md5_code,True,extracted]
        else:
            print(videos_sub1_dir, '是不包含图片的文件夹')
            for video_name in videos_files:
                video_path = os.path.join(videos_sub1_dir,video_name)
                video_part_path = video_path.split('row')[1]
                print('video_part_path', video_part_path)
                if video_part_path in videos_path_md5_json.keys():
                    print('已包含该视频',video_path)
                else:
                    md5_code =get_md5_code_str(video_part_path)
                    extracted = True if md5_code in video_part_path_md5 else False
                    # 视频路径：[md5code 有图片  已提取]
                    videos_path_md5_json[video_part_path]=[md5_code,False,extracted]

    with open(base_config.videos_path_md5_jsonpath,'w') as f:
        json.dump(videos_path_md5_json,f)

import shutil

## 修改videos 路径重新生成MD5值但是保持已提取不变
def change_video_path_and_md5():
    with open(base_config.videos_path_md5_jsonpath,'r') as f:
        videos_path_md5_json = json.load(f)
    new_json={}
    ### 对所有视频路径进行替换
    for video_path,(md5_code,has_image,extracted) in videos_path_md5_json.items():
        print(video_path.split('row'))
        new_path = video_path.split('row')[1]
        new_md5_code = get_md5_code_str(new_path)
        new_json[new_path]=[new_md5_code,has_image,extracted]
        ## 对所有已提取的文件名进行替换
        # src_file_dir = os.path.join(r'D:\dataset\crawler\extract_face',md5_code)
        # dst_file_dir = os.path.join(r'D:\dataset\crawler\extract_face',new_md5_code)
        # shutil.move(src_file_dir,dst_file_dir)
    with open(base_config.videos_path_md5_jsonpath,'w') as f:
        json.dump(new_json,f)





# 使用md5
def vaild_same_video_md5(video_path=None):
    if video_path:
        video_path=video_path
    else:
        video_path=r'D:\dataset\crawler\row\videos\003-1\003-1.mp4'
    f = open(video_path, 'rb')
    md5_obj = hashlib.md5()
    while True:
        d = f.read(8096)
        if not d:
            break
        md5_obj.update(d)
    hash_code = md5_obj.hexdigest()
    f.close()
    md5 = str(hash_code).lower()
    return md5

# 验证视频内容是否重复
def vaild_all_video():
    # md5内容映射表验证所有视频是否内容重复
    md5_content_dict = {}
    videos_dir = os.path.join(base_config.row_dir, 'videos')
    videos_dirname_lists = os.listdir(videos_dir)
    for videos_dirname in tqdm(videos_dirname_lists):
        # 循环videos文件夹
        videos_sub1_dir = os.path.join(videos_dir, videos_dirname)
        if os.path.isdir(videos_sub1_dir):  # 若是目录
            videos_files = glob.glob(os.path.join(videos_sub1_dir, "*.mp4")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.avi")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.mpg")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.webm")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.mpeg"))
            if videos_files is [] or len(videos_files) == 0:
                print('该路径下没有视频', videos_sub1_dir)
                continue

            for video_name in videos_files:
                video_path = os.path.join(videos_sub1_dir, video_name)
                # print('video_path', video_path)
                video_content_md5 = vaild_same_video_md5(video_path)
                # print('video_content_md5',video_content_md5)
                if video_content_md5 in md5_content_dict.keys():
                    print('已包含该视频')
                    print('##################source',md5_content_dict[video_content_md5])
                    print('##################repeat',video_path)

                else:
                    md5_content_dict[video_content_md5] = video_path

        elif os.path.isfile(videos_sub1_dir):
            video_path = videos_sub1_dir
            print('video_path', video_path)
            video_content_md5 = vaild_same_video_md5(video_path)
            if video_content_md5 in md5_content_dict:
                print('已包含该视频')
                print('##################source', md5_content_dict[video_content_md5])
                print('##################repeat', video_path)
            else:
                md5_content_dict[video_content_md5] = video_path


if __name__=='__main__':
    generate_videos_path_map(reload_all=True)
    # generate_images_path_map(False)
    # print(vaild_same_video_md5(r'D:\dataset\crawler\row\videos\001-1\001-1.mp4'))
    # vaild_all_video()
    # change_video_path_and_md5()