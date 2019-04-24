import numpy as np
import gist
import cv2


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


a1 = np.load("embedding/s2/1.npy")[0]
a2 = np.load("embedding/s2/2.npy")[0]
b1 = np.load("embedding/s3/7.npy")[0]
b2 = np.load("embedding/s3/9.npy")[0]

print("a1-a2: {:.4f}".format(dist(a1, a2)))
print("b1-b2: {:.4f}".format(dist(b1, b2)))
print("a1-b1: {:.4f}".format(dist(a1, b1)))
print("a2-b2: {:.4f}".format(dist(a2, b2)))

print()

a1 = gist.extract(cv2.imread("data_align/s2/1.pgm"))
a2 = gist.extract(cv2.imread("data_align/s2/2.pgm"))
b1 = gist.extract(cv2.imread("data_align/s3/7.pgm"))
b2 = gist.extract(cv2.imread("data_align/s3/9.pgm"))

print("a1-a2: {:.4f}".format(dist(a1, a2)))
print("b1-b2: {:.4f}".format(dist(b1, b2)))
print("a1-b1: {:.4f}".format(dist(a1, b1)))
print("a2-b2: {:.4f}".format(dist(a2, b2)))
