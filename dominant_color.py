import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("dat/cropped/1.jpg")

# average
average = img.mean(axis=0).mean(axis=0)

#k-means clustering
pixels = np.float32(img.reshape(-1, 3))
n_colors = 5
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv.KMEANS_RANDOM_CENTERS

_, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)

dominant = palette[np.argmax(counts)]

print(f"average\n{average}")
print(f"dominant\n{dominant}")

avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

indices = np.argsort(counts)[::-1]   
freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
rows = np.int_(img.shape[0]*freqs)

dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
for i in range(len(rows) - 1):
    dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

dom_patch_test = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(dominant)
    
cv.imshow("average", avg_patch)
cv.imshow("dominant", dom_patch_test)
cv.waitKey(0)