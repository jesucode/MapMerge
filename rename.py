import os

for i in range(43, 34, -1):
    old_name = "file_" + str(i)
    new_name = "file_" + str(i - 1)
    os.rename(old_name, new_name)


def merge_images_with_orb(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Find ORB keypoints and descriptors in each image
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Find matching keypoints using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate transformation matrix using the first 10 matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the second image to align it with the first image
    h2, w2 = img2.shape[:2]
    warped_img2 = cv2.warpPerspective(img2, M, (w2, h2))

    # Blend the two images together
    h1, w1 = img1.shape[:2]
    blended_img = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
    blended_img[:, :w1] = img1
    warped_img2 = cv2.resize(warped_img2, (w1, h1))  # Resize warped_img2 to match size of img1
    blended_img[:, w1:] = warped_img2

    # Uncomment the following line to show the matched keypoints on the blended image
    # blended_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], blended_img, flags=2)

    return blended_img