import cv2
import numpy as np

# Define the path to the input images
input_path = "./f/static_frame_{}.png"


# Load the two images
img1 = cv2.imread("./f/static_frame_8.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./f/static_frame_9.jpg", cv2.IMREAD_GRAYSCALE)

# Detect features and compute descriptors in both images
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create a matcher object
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the descriptors in both images
matches = matcher.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Select the top 10% matches
matches = matches[:int(len(matches) * 0.1)]

# Extract the matched keypoints in both images
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the perspective transformation matrix using the matched keypoints
M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
M = np.float32(M)

# Warp the second image using the transformation matrix
result = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

# Copy the first image onto the result image
result[0:img1.shape[0], 0:img1.shape[1]] = img1

# Save the merged image
cv2.imwrite("merged_image.png", result)