import cv2
import numpy as np

# Define the path to the input images
input_path = "./f/static_frame_{}.png"

start_i =  7
end_i = 18
# Load the first two images
img1 = cv2.imread(input_path.format(start_i), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(input_path.format(start_i+1), cv2.IMREAD_GRAYSCALE)

# Merge the first two images
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matches = matcher.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:int(len(matches) * 0.1)]
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
M = np.float32(M)

result = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img1.shape[0], 0:img1.shape[1]] = img1

# Merge the remaining images
for i in range(start_i+2, end_i):
    img = cv2.imread(input_path.format(i), cv2.IMREAD_GRAYSCALE)
    kp1, des1 = orb.detectAndCompute(result, None)
    kp2, des2 = orb.detectAndCompute(img, None)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.1)]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    M = np.float32(M)

    img_warped = cv2.warpPerspective(img, M, (result.shape[1] + img.shape[1], result.shape[0]))
    img_warped[0:result.shape[0], 0:result.shape[1]] = result
    result = img_warped

# Save the final merged image
cv2.imwrite("merged_image.png", result)