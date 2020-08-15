import numpy as np
import cv2 as cv
from sklearn import cluster
import matplotlib.pyplot as plt


def compress_image(img, num_clusters):
    """对图片进行聚类压缩"""
    X = img.reshape((-1, 1))
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=4, random_state=6)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)

    return input_image_compressed


def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.show()


input_image = "data/flower_image.jpg"
num_bits = 3
num_clusters = 2 ** num_bits
compressed_ratio = round(100 * (8 - num_bits) / 8, 2)
print("Compression rate = " + str(compressed_ratio) + "%")

image = cv.imread(input_image, 0)
plot_image(image, "Original Image")

plot_image(compress_image(image, num_clusters), "Compressed Image")