import os
import sys
import uuid
from base_config import row_dir,uuid_videos_path_jsonpath,videos_path_uuid_jsonpath,data_rootdir
import base_config
import glob
import json
from tqdm import tqdm
import hashlib
"""
统计文件夹路径和图片信息
"""



def make_file_id(src):
    m1 = hashlib.md5()
    m1.update(src.encode('utf8'))
    return m1.hexdigest()

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



def generate_videos_path_map(reload_all=False):
    """
    生成视频数据地址对应uuid_code 是否包含图片  是否已提取人脸
    video_path:[uuid_code,have_images,extracted]
    :param updata:
    :return:
    """
    if reload_all:
        os.remove(videos_path_uuid_jsonpath)
        os.remove(uuid_videos_path_jsonpath)
    try:
        with open(videos_path_uuid_jsonpath,'r') as f0:
            videos_path_uuid = json.load(f0)
        with open(uuid_videos_path_jsonpath,'r') as f1:
            uuid_videos_path = json.load(f1)
    except Exception:
        videos_path_uuid,uuid_videos_path = {},{}
    assert 'videos' in os.listdir(row_dir), 'config 中文件夹内不包含videos 文件夹'

    videos_dir = os.path.join(row_dir, 'videos')

    videos_dirname_lists = os.listdir(videos_dir)

    for videos_dirname in tqdm(videos_dirname_lists):
        # 循环videos文件夹
        videos_sub1_dir = os.path.join(videos_dir, videos_dirname)
        if os.path.isdir(videos_sub1_dir):  # 若是目录
            images_files = glob.glob(os.path.join(videos_sub1_dir, "*.png")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.jpg")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.bmp"))
            videos_files = glob.glob(os.path.join(videos_sub1_dir, "*.mp4")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.avi")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.mpg")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.webm")) + \
                           glob.glob(os.path.join(videos_sub1_dir, "*.mpeg"))

            if videos_files is [] or len(videos_files) ==0:
                print('videos_sub1_dir',videos_sub1_dir)
                raise Exception('videos sub file do not have any videos like mp4 avi mpg mpeg wmf')
            if images_files is not [] and len(images_files)!=0:
                print(videos_sub1_dir,'是包含图片的文件夹')
                for video_name in videos_files:

                    video_path = os.path.join(videos_sub1_dir,video_name)
                    print('video_path', video_path)
                    if video_path in videos_path_uuid.keys():
                        print('已包含该视频')
                        uuid_code = videos_path_uuid[video_path][0]
                        videos_path_uuid.update({video_path:[uuid_code,True,False]})
                        uuid_videos_path.update({uuid_code: video_path})
                    else:
                        uuid_code = str(uuid.uuid4())
                        videos_path_uuid.update({video_path:[uuid_code,True,False]})
                        uuid_videos_path.update({uuid_code: video_path})
            else:
                print(videos_sub1_dir, '是不包含图片的文件夹')
                for video_name in videos_files:

                    video_path = os.path.join(videos_sub1_dir,video_name)
                    print('video_path', video_path)
                    if video_path in videos_path_uuid.keys():
                        print('已包含该视频')
                        uuid_code = videos_path_uuid[video_path][0]
                        videos_path_uuid.update({video_path:[uuid_code,False,False]})
                        uuid_videos_path.update({uuid_code: video_path})
                    else:
                        uuid_code = str(uuid.uuid4())
                        videos_path_uuid.update({video_path:[uuid_code,False,False]})
                        uuid_videos_path.update({uuid_code: video_path})

        elif os.path.isfile(videos_sub1_dir):
            print(videos_sub1_dir,'是videos目录下直接的视频文件')
            uuid_code = str(uuid.uuid4())
            add_map = {videos_sub1_dir: [uuid_code, False]}
            videos_path_uuid.update(add_map)
            uuid_videos_path.update({uuid_code: videos_sub1_dir})

    with open(videos_path_uuid_jsonpath,'w') as f:
        json.dump(videos_path_uuid,f)
    with open(uuid_videos_path_jsonpath,'w') as f:
        json.dump(uuid_videos_path,f)

if __name__=='__main__':
    # generate_videos_path_map(False)
    generate_images_path_map(False)