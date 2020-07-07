# coding=utf-8
import os, cv2, numpy
import numpy as np
import logging
import copy

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

imgSize = [112, 96];
coord5point = [[30.2946, 51.6963],
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]

face_landmarks = [[325.014, 151.109],
                  [400.579, 168.538],
                  [365.502, 204.401],
                  [315.705, 240.369],
                  [369.122, 251.158]]

face_landmarks68 = [[282, 156], [278, 176], [277, 198], [276, 219],
                    [279, 241], [287, 262], [299, 280], [315, 293],
                    [333, 300], [352, 300], [372, 293], [391, 282],
                    [406, 270], [418, 253], [426, 236], [434, 217],
                    [439, 197], [301, 138], [313, 128], [328, 127],
                    [343, 130], [356, 138], [384, 144], [400, 142],
                    [417, 145], [431, 156], [437, 171], [365, 160],
                    [362, 174], [359, 188], [357, 202], [337, 211],
                    [344, 215], [352, 219], [360, 219], [368, 218],
                    [312, 152], [322, 147], [332, 149], [340, 159],
                    [330, 159], [320, 157], [385, 169], [396, 164],
                    [407, 167], [414, 176], [405, 177], [394, 174],
                    [317, 242], [329, 237], [341, 235], [347, 238],
                    [356, 237], [364, 243], [371, 253], [361, 258],
                    [351, 259], [343, 259], [335, 257], [326, 251],
                    [322, 242], [339, 243], [346, 245], [354, 245],
                    [366, 251], [353, 247], [345, 247], [338, 245]]


def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    print(M[:2])
    return dst, M[:2]


def draw_landmark(img_im, land):
    img = copy.deepcopy(img_im)
    n = np.array(land)
    for i in range(len(land)):
        cv2.circle(img, (int(n[i][0]), int(n[i][1])), 2, (0, 0, 255), -1)
    cv2.imshow('aaa', img)
    # cv2.imwrite('land5.jpg',img)
    # cv2.waitKey(0)


def draw_landmark_warpAffine(img, land, M):
    new_n = []
    for i in range(len(land)):
        pts = []
        pts.append(np.squeeze(np.array(M[0]))[0] * land[i][0] + np.squeeze(np.array(M[0]))[1] * land[i][1] +
                   np.squeeze(np.array(M[0]))[2])
        pts.append(np.squeeze(np.array(M[1]))[0] * land[i][0] + np.squeeze(np.array(M[1]))[1] * land[i][1] +
                   np.squeeze(np.array(M[1]))[2])
        new_n.append(pts)
    n = np.array(new_n)
    for i in range(len(land)):
        cv2.circle(img, (int(n[i][0]), int(n[i][1])), 2, (0, 0, 255), -1)
    cv2.imshow('bbb', img)
    # cv2.imwrite('land68.jpg',img)
    # cv2.waitKey(0)


def main():
    pic_path = r'XZQ.jpg'
    img_im = cv2.imread(pic_path)
    draw_landmark(img_im, face_landmarks)
    # cv2.imshow('affine_img_im', img_im)
    dst, M = warp_im(img_im, face_landmarks, coord5point)
    cv2.imshow('affine', dst)
    # cv2.imwrite('affine.jpg',dst)
    draw_landmark_warpAffine(dst, face_landmarks68, M)
    crop_im = dst[0:imgSize[0], 0:imgSize[1]]
    cv2.imshow('affine_crop_im', crop_im)
    # cv2.imwrite('affine_crop_im.jpg',crop_im)


if __name__ == '__main__':
    main()
    cv2.waitKey()
    pass
