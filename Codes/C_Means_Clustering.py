#C Means Clustering Classifier

#Omar Mostafa Hosny        - 16P8170
#Laila Ayman               - 16P3084
#Ahmed Abo Alhagag Ahmed   - 16p6061
#Abdelrahman Ayman Mohamed - 16P8069

import cv2
import numpy as np

# Read image
image = cv2.imread("/home/fadlo/My_Projects/Machine_Vision/Projects/Project_1/project01/TestingImages/3096.jpg")

#Inicilized Ms
M1 = np.random.randint(100, size=(3,1)) + 1      #To avoid getting 0
M2 = np.random.randint(100, size=(3,1)) + 1

img_R = image[:,:,0]
img_G = image[:,:,1]
img_B = image[:,:,2]

for epoch in range(5):

    #Distance
    distance_to_m1 = np.sqrt((img_R - M1[0])**2 + (img_G - M1[1])**2 + (img_B - M1[2])**2)
    distance_to_m2 = np.sqrt((img_R - M2[0])**2 + (img_G - M2[1])**2 + (img_B - M2[2])**2)

    x = np.zeros((distance_to_m1.shape[0], distance_to_m1.shape[1]))

    total_R_group1 = 0
    total_G_group1 = 0
    total_B_group1 = 0
    counter_group1 = 0
    total_R_group2 = 0
    total_G_group2 = 0
    total_B_group2 = 0
    counter_group2 = 0

    for i in range(distance_to_m1.shape[0]):
        for j in range(distance_to_m1.shape[1]):
            if distance_to_m1[i][j] > distance_to_m2[i][j]:
                total_R_group1 = total_R_group1 + image[i,j,0]
                total_G_group1 = total_G_group1 + image[i,j,1]
                total_B_group1 = total_B_group1 + image[i,j,2]
                counter_group1 = counter_group1 + 1

            else:
                total_R_group2 = total_R_group2 + image[i,j,0]
                total_G_group2 = total_G_group2 + image[i,j,1]
                total_B_group2 = total_B_group2 + image[i,j,2]
                counter_group2 = counter_group2 + 1
    
    M1[0] = total_R_group1 / counter_group1
    M1[1] = total_G_group1 / counter_group1
    M1[2] = total_B_group1 / counter_group1

    M2[0] = total_R_group2 / counter_group2
    M2[1] = total_G_group2 / counter_group2
    M2[2] = total_B_group2 / counter_group2

    for i in range(distance_to_m1.shape[0]):
        for j in range(distance_to_m1.shape[1]):
            if distance_to_m1[i][j] < distance_to_m2[i][j]:
                x[i,j] = 255

print("M1 is : ", M1)
print("M2 is : ", M2)
cv2.imshow("Original Image", image)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if x[i,j] == 255:
            image[i,j,0] = M2[0]
            image[i,j,1] = M2[1]
            image[i,j,2] = M2[2]
        else:
            image[i,j,0] = M1[0]
            image[i,j,1] = M1[1]
            image[i,j,2] = M1[2]

cv2.imshow("Segmented with C-Means Clustering", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
