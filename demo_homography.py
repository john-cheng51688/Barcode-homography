#example:  $ python3 demo_homography.py 5451_big.jpg
import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

def mouseHandler(event,x,y,flags,param):
    global im_temp, pts_dst

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im_temp,(x,y),1,(0,255,255),5,cv2.LINE_AA)
        cv2.imshow("Image", im_temp)
        if len(pts_dst) < 4:
        	pts_dst = np.append(pts_dst,[(x,y)],axis=0)

#===============================================
#		image insert
#===============================================
img_combine = np.zeros((800,1000,3), np.uint8)
img_combine[:,0:1000] = (255,255,255)


# Destination image
img_src = cv2.imread(sys.argv[1])

# Create a window
cv2.namedWindow("Image", 1)

im_temp = img_src
pts_dst = np.empty((0,2),dtype=np.int32)

cv2.setMouseCallback("Image",mouseHandler)

cv2.imshow("Image", im_temp)
cv2.waitKey(0)

print("pts_dst")
print(pts_dst)

pts1 = np.float32(pts_dst)


dst_xy_min = np.min(pts1,axis = 0)
dst_xy_max = np.max(pts1,axis = 0)
dst_xy_width = dst_xy_max[0] - dst_xy_min[0]
dst_xy_hight = dst_xy_max[1] - dst_xy_min[1]


print("dst_xy_min",dst_xy_min)
print("dst_xy_max",dst_xy_max)


pts2= np.float32([[dst_xy_min[0],dst_xy_min[1]],  [dst_xy_max[0],dst_xy_min[1]],  [dst_xy_max[0],dst_xy_max[1]],  [dst_xy_min[0],dst_xy_max[1]]])


M = cv2.getPerspectiveTransform(pts1,pts2)


offsetSize = int(np.min([dst_xy_width,dst_xy_hight])/3)+10
transformed = np.zeros((int(dst_xy_width+offsetSize), int(dst_xy_hight+offsetSize)), dtype=np.uint8);
img_dst = cv2.warpPerspective(img_src, M, transformed.shape,borderValue=(255, 255, 255))

plt.subplot(121),plt.imshow(img_src),plt.title('Before')
plt.subplot(122),plt.imshow(img_dst),plt.title('After')

#cv2.imwrite("n1_img.jpg", dst)
plt.show()


