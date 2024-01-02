import cv2
import numpy as np

import cv2
import numpy as np

def merge_images_with_orb_color(img_path1, img_path2, fac):
    # Load the input images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Convert the input images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB feature detector
    
    feat =10000000

    orb = cv2.ORB_create(nfeatures=  feat , patchSize=64 )

    # Find the key points and descriptors for each image
    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)

    # Initialize the Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors in both images
    matches = bf.match(desc1, desc2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate the size of the output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out_h = max(h1, h2)
    out_w = w1 + w2

    # Find the homography matrix based on the matched features
    midpoint = int(w1/2)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Apply the perspective transformation to the second image
    warped_img2 = cv2.warpPerspective(img2, M, (out_w, out_h))

    # Create the output image by blending the input images together
    blended_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    blended_img[:h1, :w1] = img1
    #blended_img = cv2.addWeighted(blended_img, 0.5, warped_img2, 0.5, 0)


    # Crop the blended image where the second image is present
    
    gray = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU)[1]
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    mask = cv2.bitwise_not(mask)


    blended_img = cv2.bitwise_and( blended_img , mask )
    blended_img = cv2.bitwise_or( blended_img , warped_img2 )

    return blended_img


first = 1
last = 100
a = 1
def mergei(i):
    
    # Construct the file paths for the two images to be merged
    img_path1 = f'./fol/static_frame_{i}.png'
    img_path2 = f'./fol/static_frame_{i+1}.png'

    # Merge the two images using the merge_images_with_orb function
    
    merged_img = merge_images_with_orb_color(img_path1, img_path2 ,  i - first)
    

    # Save the resulting merged image with a filename indicating which files were merged
    output_filename = f'./fol/mer2/merged_{i}.png'
    cv2.imwrite(output_filename, merged_img)
    print(  f' saving  : merged_{i}.png  {i} ,{i +1}')


#use first arg as value for i
import sys
mergei(int(sys.argv[1]))




