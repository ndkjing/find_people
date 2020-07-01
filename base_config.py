import os

# 项目根路径
# root_dir  = r'D:\PythonProject\find_people'
root_dir  = r'C:\PythonProject\find_people'


## 数据路径
# data_rootdir = 'D:\dataset\crawler'
data_rootdir = 'C:\PythonProject\dataset\crawler'
row_dir = os.path.join(data_rootdir,'row')
extract_face_dir = os.path.join(data_rootdir,'extract_face')
extrace_posture_dir = os.path.join(data_rootdir,'extract_posture')
extrace_voice_dir = os.path.join(data_rootdir,'extract_voice')

# json映射文件路径
uuid_videos_path_jsonpath = os.path.join(root_dir,r'data\uuid_videos_path.json')
videos_path_uuid_jsonpath = os.path.join(root_dir,r'data\videos_path_uuid.json')

md5_images_path_jsonpath = os.path.join(root_dir,r'data\md5_images_path.json')
images_path_md5_jsonpath = os.path.join(root_dir,r'data\images_path_md5.json')

# 视频检测间隔帧数量
frame_interval = 5

# 训练模型路径
trained_model = r"D:\PythonProject\find_people\core\face\face_detect_lightdsfd\weights\dsfdv2_r18.pth"

# 扩大脸部检测框来保存，扩大比例
face_enlarge = 1.5

# 使用GPU
use_cuda = True
# import  torch
# print(torch.cuda.is_available())