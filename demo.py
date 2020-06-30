import cv2
#
# cap = cv2.VideoCapture(r'D:\dataset\crawler\row\videos\(wu夜XUN花)高质量高价\wu夜XUN花 2020-03-16 21-14-19-212_(new).avi')
# i = 0
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(fps)
# cap.set(cv2.CAP_PROP_FPS,2)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(fps)
# while True:
#     flag,frame = cap.read()
#     if not flag:
#         print('video is over')
#         break
#     cv2.imshow('frame',frame)
#     cv2.waitKey(1)
# cap.release()
# cv2.destroyAllWindows()
file_path=r'D:\dataset\crawler\row\images\actors\刘琳\ia_10026.jpg'
# i =cv2.imread()
# print(i)
# import cv2
import numpy as np

cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
cv2.imencode('.jpg', cv_img)[1].tofile('我.jpg')   # 写入中文路径