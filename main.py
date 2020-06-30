import json
import cv2

from base_config import uuid_videos_path_jsonpath,videos_path_uuid_jsonpath,frame_interval




def extract_face():

    ### 提取视频中人脸
    with open(videos_path_uuid_jsonpath,'r') as f:
        videos_path_uuid = json.load(f)

    for video_path,(uuid_code,have_images) in videos_path_uuid.items():
        print(video_path,uuid_code,have_images)
        cap = cv2.VideoCapture(video_path)
        fram_count = 0
        while True:
            flag, frame = cap.read()
            if not flag:
                print('video is over ')
                break
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        cv2.release()


if __name__=="__main__":
    extract_face()














