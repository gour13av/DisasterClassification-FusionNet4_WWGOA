import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops


# Texture Feature
def TextureFeature(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    co_matrix = greycomatrix(img, [5], [0], 256,True, True)

    contrast = greycoprops(co_matrix, 'contrast').ravel()
    correlation = greycoprops(co_matrix, 'correlation').ravel()
    energy = greycoprops(co_matrix, 'energy').ravel()
    homogeneity = greycoprops(co_matrix, 'homogeneity').ravel()
    dissimilarity = greycoprops(co_matrix, 'dissimilarity').ravel()

    textureFeature = np.concatenate([contrast, correlation, energy, homogeneity, dissimilarity])
    return textureFeature


# # Color Feature
# def ColorFeature(image):
#     # color Histogram Feature
#     hist = cv2.calcHist([image], [0,1,2], None, [10, 10, 10], [0, 256, 0, 256, 0, 256])
#     # color moments Feature
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     moments = cv2.moments(img)
#     colorFeatures = np.concatenate([hist.flatten(), np.array(list(moments.values()))])
#     return colorFeatures


# Geometric Feature
def GeometricFeature(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geometric_feat = []
    # Calculate contour-based features for each contour
    for contour in contours:
        # Area
        area = cv2.contourArea(contour)
        if area == 0.0:
            continue
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        # Compactness
        compactness = (perimeter ** 2) / (4 * 3.1415 * area)
        feat = np.concatenate([np.atleast_1d(area, perimeter, compactness)])
        geometric_feat.extend(feat.flatten())
        return geometric_feat

# Shape Feature
def ShapeFeatures(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_feat = []
    # Calculate shape features for each contour
    for i, contour in enumerate(contours):
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Skip small contours
        if area < 50:
            continue

        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Calculate rectangularity
        x, y, w, h = cv2.boundingRect(contour)
        rectangularity = area / (w * h)

        # Calculate solidity
        hull = cv2.convexHull(contour)
        solidity = area / cv2.contourArea(hull)

        feat = np.concatenate([np.atleast_1d(circularity, rectangularity, solidity)])
        shape_feat.extend(feat.flatten())
        return shape_feat


def feature_extraction(image):
    texture_feat = TextureFeature(image)
    # color_feat = ColorFeature(image)
    geometric_feat = GeometricFeature(image)
    shape_feat = ShapeFeatures(image)
    feature = np.concatenate([texture_feat, geometric_feat, shape_feat])
    return feature