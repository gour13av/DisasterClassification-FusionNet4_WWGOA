import cv2
import os
os.makedirs('./Results', exist_ok=True)
os.makedirs('./Saved data', exist_ok=True)
os.makedirs('./Pictorial Results', exist_ok=True)
from WOWWP import Wowwp
from save_load import save, load
from unet import dilated_unet
from Objective_function import fitness
import numpy as np
from PIL import Image
from feature_extraction import feature_extraction
from sklearn.model_selection import train_test_split


def datagen():
    BaseDir = './Dataset/RescueNet/'
    imgResultPath = './Pictorial Results/'
    FolderDir = os.listdir(BaseDir)
    n = 0
    label = []
    features = []
    lb = [2]
    ub = [7]
    pop_size = 6
    prob_size = len(lb)
    maximum_iteration = 100
    best_solution = Wowwp(fitness, lb, ub, pop_size, prob_size, maximum_iteration)
    save('unet_best_solution', best_solution)
    best_solution = load('unet_best_solution')
    input_shape = (200, 200, 3)
    # create segmentation model
    segmentation_model = dilated_unet(input_shape, best_solution)
    segmentation_model.save('segmentation_model.h5')

    for folder in FolderDir:
        imageDir = os.listdir(BaseDir + folder)
        for img in imageDir:
            image = cv2.imread(BaseDir + folder + '/' + img)
            image = cv2.resize(image, dsize=(200, 200), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(imgResultPath + '1.Original image.png', image)

            # preprocessing
            # Noise Reduction - Bi-Lateral Filter
            DenoisedImg = cv2.bilateralFilter(image, 15, 17, 17)
            # cv2.imwrite(imgResultPath + '2.Denoised img.png', DenoisedImg)

            # Enhancement Technique - CLAHE(Histogram Equalization)
            lab_img = cv2.cvtColor(DenoisedImg, cv2.COLOR_BGR2Lab)
            l_channel, a, b = cv2.split(lab_img)
            # apply CLAHE to l_channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            EnhancedImg = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
            # cv2.imwrite(imgResultPath + '3.Enhanced Image.png', EnhancedImg)

            # segmentation
            preprocessed_image = np.expand_dims(EnhancedImg, axis=0)
            predicted_mask = segmentation_model.predict(preprocessed_image)

            # Threshold the mask to get binary values (0 or 1)
            threshold = 0.8
            binary_mask = (predicted_mask > threshold).astype(np.uint8)

            # Convert the binary mask to a segmented image
            segmented_image = Image.fromarray(binary_mask[0, :, :, 0] * 255)

            # # Save or display the segmented image
            # segmented_image.save('segmented_image.png')
            # segmented_image.show()
            segmented_image_array = np.array(segmented_image)

            # Resize the segmented image array to match the size of the original image
            segmented_image_array = cv2.resize(segmented_image_array,
                                               (preprocessed_image.shape[2], preprocessed_image.shape[1]))

            # Create a copy of the original image
            highlighted_image = preprocessed_image.copy()

            # Squeeze the binary mask to remove the singleton dimension
            binary_mask_squeezed = np.squeeze(binary_mask, axis=0)
            segmented_image_array = cv2.cvtColor(segmented_image_array, cv2.COLOR_GRAY2RGB)
            # Apply the segmented image as a mask to highlight the segmented part
            for channel in range(preprocessed_image.shape[3]):
                highlighted_image[0, :, :, channel][binary_mask_squeezed[:, :, 0] > 0] = \
                    segmented_image_array[:, :, channel][binary_mask_squeezed[:, :, 0] > 0]

            # Convert the result back to PIL Image
            highlighted_image_pil = Image.fromarray(np.uint8(np.squeeze(highlighted_image)))
            # Save or display the highlighted image
            highlighted_image_pil.save(imgResultPath + '4.segmented image.png')
            segmented_image = np.array(highlighted_image_pil)
            feature = feature_extraction(segmented_image)
            features.append(feature)
            label.append(n)
        n += 1
    features = np.array(features)
    labels = np.array(label)

    # normalization
    features = features / np.max(features, axis=0)
    # absolute
    features = abs(features)
    # nan to num
    features = np.nan_to_num(features)

    learning_rate = [0.7, 0.8]  # train size
    for learn_rate in learning_rate:
        x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=learn_rate)
        save('x_train_' + str(learn_rate*100), x_train)
        save('x_test_' + str(learn_rate* 100), x_test)
        save('y_train_' + str(learn_rate*100), y_train)
        save('y_test_' + str(learn_rate*100), y_test)
