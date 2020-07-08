"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

net_type='RFB'  #The network architecture ,optional: RFB (higher precision) or slim (faster)
input_size=480   #  help='define network input size,default optional value 128/160/320/480/640/1280')
threshold=0.7    #  help='score threshold')
candidate_size=1000  #   help='nms candidate size')

test_device="cpu"  #  help='cuda:0 or cpu')


define_img_size(input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/voc-model-labels.txt"

net_type =net_type

# cap = cv2.VideoCapture(args.video_path)  # capture from video
cap = cv2.VideoCapture(0)  # capture from camera

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = test_device

candidate_size = candidate_size
threshold = threshold

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

timer = Timer()
sum = 0
def detect_image(image):
    if isinstance(image,str):
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f" {probs[i]:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
        # cv2.putText(orig_image, label,
        #             (box[0], box[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,  # font scale
        #             (0, 0, 255),
        #             2)  # line type
    print(image.shape)
    cv2.imshow('annotated', image)
    cv2.waitKey(0)


cv2.destroyAllWindows()

if __name__=='__main__':
    detect_image(r'C:\PythonProject\find_people\face.jpg')
