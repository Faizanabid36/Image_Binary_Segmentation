#KNN Classifier

#Omar Mostafa Hosny        - 16P8170
#Laila Ayman               - 16P3084
#Ahmed Abo Alhagag Ahmed   - 16p6061
#Abdelrahman Ayman Mohamed - 16P8069

import cv2
import numpy as np

# Read image
image = cv2.imread("/home/fadlo/My_Projects/Machine_Vision/Projects/Project_1/project01/TestingImages/210088.jpg")
image_R = image[:,:,0]
image_G = image[:,:,1]
image_B = image[:,:,2]


# Select ROI
r_class1 = cv2.selectROI(image)
img1_class1 = image[int(r_class1[1]):int(r_class1[1]+r_class1[3]), int(r_class1[0]):int(r_class1[0]+r_class1[2])]

r_class1 = cv2.selectROI(image)
img2_class1 = image[int(r_class1[1]):int(r_class1[1]+r_class1[3]), int(r_class1[0]):int(r_class1[0]+r_class1[2])]

r_class1 = cv2.selectROI(image)
img3_class1 = image[int(r_class1[1]):int(r_class1[1]+r_class1[3]), int(r_class1[0]):int(r_class1[0]+r_class1[2])]

r_class2 = cv2.selectROI(image)
img1_class2 = image[int(r_class2[1]):int(r_class2[1]+r_class2[3]), int(r_class2[0]):int(r_class2[0]+r_class2[2])]

r_class2 = cv2.selectROI(image)
img2_class2 = image[int(r_class2[1]):int(r_class2[1]+r_class2[3]), int(r_class2[0]):int(r_class2[0]+r_class2[2])]

r_class2 = cv2.selectROI(image)
img3_class2 = image[int(r_class2[1]):int(r_class2[1]+r_class2[3]), int(r_class2[0]):int(r_class2[0]+r_class2[2])]

#-----------------------------------------------------
#Calculate Mean R,G and B for each cropped image
class1_m1_R = np.sum(img1_class1[:,:,0]) / (img1_class1.shape[0] * img1_class1.shape[1])
class1_m1_G = np.sum(img1_class1[:,:,1])/ (img1_class1.shape[0] * img1_class1.shape[1])
class1_m1_B = np.sum(img1_class1[:,:,2]) / (img1_class1.shape[0] * img1_class1.shape[1])
class1_m1 = np.array([class1_m1_R, class1_m1_G, class1_m1_B])
class1_m1 = np.reshape(class1_m1,(3,1))

class1_m2_R = np.sum(img2_class1[:,:,0]) / (img2_class1.shape[0] * img2_class1.shape[1])
class1_m2_G = np.sum(img2_class1[:,:,1])/ (img2_class1.shape[0] * img2_class1.shape[1])
class1_m2_B = np.sum(img2_class1[:,:,2]) / (img2_class1.shape[0] * img2_class1.shape[1])
class1_m2 = np.array([class1_m2_R, class1_m2_G, class1_m2_B])
class1_m2 = np.reshape(class1_m2,(3,1))

class1_m3_R = np.sum(img3_class1[:,:,0]) / (img3_class1.shape[0] * img3_class1.shape[1])
class1_m3_G = np.sum(img3_class1[:,:,1])/ (img3_class1.shape[0] * img3_class1.shape[1])
class1_m3_B = np.sum(img3_class1[:,:,2]) / (img3_class1.shape[0] * img3_class1.shape[1])
class1_m3 = np.array([class1_m3_R, class1_m3_G, class1_m3_B])
class1_m3 = np.reshape(class1_m3,(3,1))

class2_m1_R = np.sum(img1_class2[:,:,0]) / (img1_class2.shape[0] * img1_class2.shape[1])
class2_m1_G = np.sum(img1_class2[:,:,1])/ (img1_class2.shape[0] * img1_class2.shape[1])
class2_m1_B = np.sum(img1_class2[:,:,2]) / (img1_class2.shape[0] * img1_class2.shape[1])
class2_m1 = np.array([class2_m1_R, class2_m1_G, class2_m1_B])
class2_m1 = np.reshape(class2_m1,(3,1))

class2_m2_R = np.sum(img2_class2[:,:,0]) / (img2_class2.shape[0] * img2_class2.shape[1])
class2_m2_G = np.sum(img2_class2[:,:,1])/ (img2_class2.shape[0] * img2_class2.shape[1])
class2_m2_B = np.sum(img2_class2[:,:,2]) / (img2_class2.shape[0] * img2_class2.shape[1])
class2_m2 = np.array([class2_m2_R, class2_m2_G, class2_m2_B])
class2_m2 = np.reshape(class2_m2,(3,1))

class2_m3_R = np.sum(img3_class2[:,:,0]) / (img3_class2.shape[0] * img3_class2.shape[1])
class2_m3_G = np.sum(img3_class2[:,:,1])/ (img3_class2.shape[0] * img3_class2.shape[1])
class2_m3_B = np.sum(img3_class2[:,:,2]) / (img3_class2.shape[0] * img3_class2.shape[1])
class2_m3 = np.array([class2_m3_R, class2_m3_G, class2_m3_B])
class2_m3 = np.reshape(class2_m3,(3,1))
#----------------------------------------------
#Calculate Distances
distance_to_class1_m1 = np.sqrt((image_R - class1_m1[0])**2 + (image_G - class1_m1[1])**2 + (image_B - class1_m1[2])**2)
distance_to_class1_m2 = np.sqrt((image_R - class1_m2[0])**2 + (image_G - class1_m2[1])**2 + (image_B - class1_m2[2])**2)
distance_to_class1_m3 = np.sqrt((image_R - class1_m3[0])**2 + (image_G - class1_m3[1])**2 + (image_B - class1_m3[2])**2)

distance_to_class2_m1 = np.sqrt((image_R - class2_m1[0])**2 + (image_G - class2_m1[1])**2 + (image_B - class2_m1[2])**2)
distance_to_class2_m2 = np.sqrt((image_R - class2_m2[0])**2 + (image_G - class2_m2[1])**2 + (image_B - class2_m2[2])**2)
distance_to_class2_m3 = np.sqrt((image_R - class2_m3[0])**2 + (image_G - class2_m3[1])**2 + (image_B - class2_m3[2])**2)
#---------------------------------------------
all_distances = np.dstack((distance_to_class1_m1, distance_to_class1_m2, distance_to_class1_m3, distance_to_class2_m1, distance_to_class2_m2, distance_to_class2_m3))
min_distance_index_3D = np.argsort(all_distances)
min_distance_index_2D = min_distance_index_3D[:,:,0]

x = np.zeros((min_distance_index_2D.shape[0], min_distance_index_2D.shape[1]))


for i in range(min_distance_index_2D.shape[0]):
        for j in range(min_distance_index_2D.shape[1]):
            if min_distance_index_2D[i][j] <= 2:
                x[i,j] = 255

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if x[i,j] == 255:
            image[i,j,0] = class1_m1[0]
            image[i,j,1] = class1_m1[1]
            image[i,j,2] = class1_m1[2]
        else:
            image[i,j,0] = class2_m1[0]
            image[i,j,1] = class2_m1[1]
            image[i,j,2] = class2_m1[2]

cv2.imshow("Segmented using NN Classifier", image)
cv2.waitKey(0)
cv2.destroyAllWindows()