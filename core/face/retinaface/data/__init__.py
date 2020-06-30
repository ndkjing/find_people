# import sys,os
# add_path = os.path.join(r'C:\PythonProject\find_people',r'core\face\retinaface')
# print(add_path)
# sys.path.append(add_path)
# sys.path.append(os.path.join(add_path,'data'))
from .wider_face import WiderFaceDetection, detection_collate
from .data_augment import *
from .config import *
