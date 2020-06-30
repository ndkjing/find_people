"""
refer https://www.pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""

import numpy as np
import argparse
import imutils
import cv2
import os
import matplotlib.pyplot as plt

# 不用图片尺寸对应不同阈值
thres_holds = {150:50,100:100,80:200,70:400}

def del_image(image_path):
    if os.path.exists(image_path):
        print('remove image',image_path)
        os.remove(image_path)
        return

def detect_blur_laplacian(image_path,show=False):
    image = cv2.imread(image_path)
    print(image_path)
    h,w = image.shape[:2]

    # 图片比例差距过大删除
    if (max(h, w) / min(h, w)) > 1.5:
        del_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print('y1 blur:', blur_var,blur_var>200)
    # 图片清晰度不够删除
    if max(h,w)>500:
        del_image(image_path)
    if max(h,w)>150 :
        thres_hold = thres_holds[150]
        if blur_var<thres_hold:
            del_image(image_path)
    elif max(h,w)>100:
        thres_hold = thres_holds[100]
        if blur_var<thres_hold:
            del_image(image_path)
    elif max(h,w)>80:
        thres_hold = thres_holds[80]
        if blur_var<thres_hold:
            del_image(image_path)
    elif max(h,w)>70:
        thres_hold = thres_holds[70]
        if blur_var<thres_hold:
            del_image(image_path)
    else:
        del_image(image_path)
    if show:
        cv2.imshow("Test Image", image)
        cv2.waitKey(0)
    return blur_var


def detect_blur_fft_image(image, size=100, thresh=10, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


def detect_blur_fft_images_dir(images_dir=None):
    if images_dir is None:
        f = r"D:\dataset\crawler\extract_face\9685d683-2269-4598-8fbe-417a538591ca"
        images = [os.path.join(f, i) for i in os.listdir(f)]
    else:
        images = [os.path.join(images_dir, i) for i in os.listdir(images_dir)]
    thresh = 20
    vis = -1
    test = -1
    for image_path in images:
        print(image_path)
        orig = cv2.imread(image_path)
        # orig = imutils.resize(orig, width=500)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # apply our blur detector using the FFT
        # blurry 小于0定于为模糊 FFT对小大图片检测都比较适合    拉普拉斯算子对小图检测结果较好与人直观感受不符合
        (mean, blurry) = detect_blur_fft_image(gray, size=60,
                                               thresh=thresh, vis=vis > 0)

        # draw on the image, indicating whether or not it is blurry
        image = np.dstack([gray] * 3)
        color = (0, 0, 255) if blurry else (0, 255, 0)
        text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
        text = text.format(mean)
        cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)
        print("[INFO] {}".format(text))

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)

        # check to see if are going to test our FFT blurriness detector using
        # various sizes of a Gaussian kernel
        if test > 0:
            # loop over various blur radii
            for radius in range(1, 30, 2):
                # clone the original grayscale image
                image = gray.copy()

                # check to see if the kernel radius is greater than zero
                if radius > 0:
                    # blur the input image by the supplied radius using a
                    # Gaussian kernel
                    image = cv2.GaussianBlur(image, (radius, radius), 0)

                    # apply our blur detector using the FFT
                    (mean, blurry) = detect_blur_fft_image(image, size=60,
                                                     thresh=thresh, vis=vis > 0)

                    # draw on the image, indicating whether or not it is
                    # blurry
                    image = np.dstack([image] * 3)
                    color = (0, 0, 255) if blurry else (0, 255, 0)
                    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
                    text = text.format(mean)
                    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
                    print("[INFO] Kernel: {}, Result: {}".format(radius, text))

                # show the image
                cv2.imshow("Test Image", image)
                cv2.waitKey(0)


def detect_blur_laplancian_images_dir(images_dir):
    for image_sub_name in os.listdir(images_dir):

        images_sub_dir = os.path.join(images_dir,image_sub_name)

        for image_name in os.listdir(images_sub_dir):
            if image_name.endswith('jpg'):
                detect_blur_laplacian(os.path.join(images_sub_dir,image_name))

if __name__ == '__main__':
    # detect_blur_fft_images_dir(images_dir=r"D:\dataset\crawler\extract_face\2f842791-b7a1-46a4-855f-a84b198f9e85")
    detect_blur_laplancian_images_dir(images_dir=r"D:\dataset\crawler\extract_face")
