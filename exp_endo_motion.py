import cv2
import cv2 as cv
import numpy as np

def test_feature() :
    img1 = cv2.imread('E:/DATASET/Camera_motion_estimation/20211026_1006_5004_c_9.22-12.29/70.jpg', 0)
    img2 = cv2.imread('E:/DATASET/Camera_motion_estimation/20211026_1006_5004_c_9.22-12.29/77.jpg', 0)

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # draw first 50 matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv2.imshow('Matches', match_img)
    cv2.waitKey()


def test_flow() :
    first_frame = cv2.imread('E:/DATASET/Camera_motion_estimation/20211026_1006_5004_c_9.22-12.29/70.jpg')
    second_frame = cv2.imread('E:/DATASET/Camera_motion_estimation/20211026_1006_5004_c_9.22-12.29/77.jpg')
    
    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally 
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(first_frame)
    
    # Sets image saturation to maximum
    mask[..., 1] = 255
    

        
        # Opens a new window and displays the input
        # frame
    cv.imshow("input", second_frame)
    
    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    gray = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)
    
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                    None,
                                    0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    
    cv.waitKey()

def test_flow2():
    pass

if __name__ == "__main__":
    test_feature()
    test_flow()