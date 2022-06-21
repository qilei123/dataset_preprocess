import os
import time
import math
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import mahotas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
bins = 8
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# This function allows to calculate optical flow trajectories
# The code also allows to specify step value. The greater the value the more sparse the calculation and visualisation
def calc_angl_n_transl(img, flow, step=8):
    
    '''
    input:
        - img - numpy array - image
        - flow - numpy array - optical flow
        - step - int - measurement of sparsity
    output:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
        - lines - list - list of actual optical flow lines (where each line represents a trajectory of 
        a particular point in the image)
    '''

    angles = []
    translation = []

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    for (x1, y1), (x2, y2) in lines:
        angle = abs(math.atan2(- int(y2) + int(y1), int(x2) - int(x1)))/np.pi# * 180.0 / np.pi #nomalize the angle
        length = math.hypot(int(x2) - int(x1), - int(y2) + int(y1))
        translation.append(length)
        angles.append(angle)
    #print(translation)
    return np.array(angles), np.array(translation), lines

# function for drawing optical flow trajectories 
def draw_flow(img, lines):
    
    '''
    input:
        - img - numpy array - image to draw on
        - lines - list - list of lines to draw
        - BGR image with visualised optical flow
    '''

    width_delay_ratio = 6
    height_delay_ratio = 5
    
    h, w = img.shape[:2]
        
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

# function that analyses optical flow information
def estimate_motion(angles, translation):
    
    '''
    Input:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
    Output:
        - ang_mode - float - mode of angles of trajectories. can be used to determine the direction of movement
        - transl_mode - float - mode of translation values 
        - ratio - float - shows how different values of translation are across a pair of frames. allows to 
        conclude about the type of movement
        - steady - bool - show if there is almost no movement on the video at the moment
    '''
    
    # Get indices of nonzero opical flow values. We'll use just them
    nonzero = np.where(translation >= 0)
    
    # Whether non-zero value is close to zero or not. Should be set as a thershold
    steady = np.mean(translation) < 0.5
    transl_mode = mode(translation)[0][0]
    ang_mode = mode(angles)[0][0]

    '''
    translation = translation[nonzero]

    transl_mode = mode(translation)[0][0]
    
    angles = angles[nonzero]
    ang_mode = mode(angles)[0][0]
    
    # cutt off twenty percent of the sorted list from both sides to get rid off outliers
    ten_percent = len(translation) // 10
    translations = sorted(translation)
    translations = translations[ten_percent: len(translations) - ten_percent]

    # cluster optical flow values and find out how different these cluster are
    # big difference (i.e. big ratio value) corresponds to panning, otherwise - trucking
    inliers = [tuple([inlier]) for inlier in translations]
    k_means = KMeans(n_clusters=1, random_state=0).fit(inliers)
    centers = sorted(k_means.cluster_centers_)
    print(centers)
    if centers[-1]>0:
        ratio = centers[0] / centers[-1]
    else:
        ratio = [0]
    '''
    ratio = [np.mean(translation)+np.mean(angles)]
    return ang_mode, transl_mode, ratio, steady

def process_video_with_opt_flow():
    # specify directory and file name 
    dir_path = "/home/qilei/DEVELOPMENT/dataset_preprocess/temp_datas/Camera_motion_estimation/"
    filename = "20211026_1006_5004_c_9.22-12.29_crop.avi"

    # initialise stream from video
    cap = cv.VideoCapture(os.path.join(dir_path, filename))
    ret, prvs = cap.read()

    # initialise video writer
    frameRate = int(cap.get(cv.CAP_PROP_FPS))
    codec = cv.VideoWriter_fourcc(*'XVID')
    save_name = os.path.join(dir_path,"motion_" + filename)
    outputStream = cv.VideoWriter(save_name, codec, frameRate, (int(cap.get(3)),int(cap.get(4))))

    # set parameters for text drawn on the frames
    font = cv.FONT_HERSHEY_COMPLEX
    fontScale = 2
    fontColor = (68, 148, 213)
    lineType  = 3

    # initialise text variables to draw on frames
    angle = 'None'
    translation = 'None'
    motion = 'None'
    motion_type = 'None'
    # set counter value
    count = 1

    # main loop
    while True:
        # read a new frame
        ret, nxt = cap.read()
        
        if not ret:
            break
        scale_percent = 50 # percent of original size
        width = int(nxt.shape[1] * scale_percent / 100)
        height = int(nxt.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        nxt = cv2.resize(nxt, dim, interpolation = cv2.INTER_AREA)        

        # if the image is colored
        if len(prvs.shape) == 3:
            prvs_gray = cv.cvtColor(prvs.copy(), cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(nxt.copy(), cv.COLOR_BGR2GRAY)
        else:
            prvs_gray = prvs.copy()
            next_gray = nxt.copy()
            
        if count > 1 :
            
            # calculate optical flow
            flow = cv.calcOpticalFlowFarneback(prvs_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #print(flow.shape)
            # calculate trajectories and analyse them
            angles, transl, lines = calc_angl_n_transl(prvs_gray, flow, step=8)
            ang_mode, transl_mode, ratio, steady = estimate_motion(angles, transl)

            # draw trajectories on the frame
            #         next_gray = draw_flow(next_gray.copy(), lines)
            next_gray = cv.cvtColor(next_gray.copy(), cv.COLOR_GRAY2BGR)

            #         angle = str(round(ang_mode, 2))
            #         translation = str(round(transl_mode, 2))
            '''
            motion = 'No motion' if steady else round(ratio[0], 8)
            if isinstance(motion, float):
                motion_type = 'Panning' if motion > 0.6 else 'Trucking'
            '''
            motion = round(ratio[0], 8)
            motion_type = "mix"
            #count = 0

        # put values on the frame
        #     cv.putText(next_gray, angle, (50,100), font, fontScale, fontColor, lineType)
        #     cv.putText(next_gray, translation, (50,150), font, fontScale, fontColor, lineType)
        cv.putText(next_gray, str(motion), (50,90), font, fontScale, fontColor, lineType)
        cv.putText(next_gray, motion_type, (50,150), font, fontScale, fontColor, lineType)
        
        # write the frame to the new video

        if not os.path.exists(save_name[:-4]):
            os.makedirs(save_name[:-4])
        print(count)
        cv.imwrite(os.path.join(save_name[:-4],str(count)+".jpg"),next_gray)

        outputStream.write(next_gray)
        
        # update the previous frame
        prvs = nxt.copy()
        count += 1

    #outputStream.release()

def process_video_with_global_feature():
    # specify directory and file name
    dir_path = "/home/qilei/DEVELOPMENT/dataset_preprocess/temp_datas/Camera_motion_estimation/"
    filename = "20211026_1006_5004_c_9.22-12.29_crop.avi"

    # initialise stream from video
    cap = cv.VideoCapture(os.path.join(dir_path, filename))
    ret, prvs = cap.read()

    # initialise video writer
    frameRate = int(cap.get(cv.CAP_PROP_FPS))
    codec = cv.VideoWriter_fourcc(*'XVID')
    save_name = os.path.join(dir_path, "gf_motion_" + filename)
    outputStream = cv.VideoWriter(save_name, codec, frameRate, (int(cap.get(3)), int(cap.get(4))))

    # set parameters for text drawn on the frames
    font = cv.FONT_HERSHEY_COMPLEX
    fontScale = 2
    fontColor = (68, 148, 213)
    lineType = 3

    # initialise text variables to draw on frames
    angle = 'None'
    translation = 'None'
    motion = 'None'
    motion_type = 'None'
    dist1 = 0
    dist2 = 0
    dist3 = 0
    # set counter value
    count = 1

    # main loop
    while True:
        # read a new frame
        ret, nxt = cap.read()

        if not ret:
            break
        scale_percent = 50  # percent of original size
        width = int(nxt.shape[1] * scale_percent / 100)
        height = int(nxt.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        nxt = cv2.resize(nxt, dim, interpolation=cv2.INTER_AREA)

        # if the image is colored
        if len(prvs.shape) == 3:
            prvs_gray = cv.cvtColor(prvs.copy(), cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(nxt.copy(), cv.COLOR_BGR2GRAY)
        else:
            prvs_gray = prvs.copy()
            next_gray = nxt.copy()

        if count > 1:


            pre_fv_hu_moments = fd_hu_moments(prvs_gray)
            pre_fv_haralick = fd_haralick(prvs_gray)
            pre_fv_histogram = fd_histogram(prvs)

            next_fv_hu_moments = fd_hu_moments(next_gray)
            next_fv_haralick = fd_haralick(next_gray)
            next_fv_histogram = fd_histogram(nxt)

            dist1 = round(np.linalg.norm(pre_fv_hu_moments - next_fv_hu_moments),8)
            dist2 = round(np.linalg.norm(pre_fv_haralick - next_fv_haralick),8)
            dist3 = round(np.linalg.norm(pre_fv_histogram - next_fv_histogram),8)

            #motion = 0
            #motion_type = "mix"


        # put values on the frame
        #     cv.putText(next_gray, angle, (50,100), font, fontScale, fontColor, lineType)
        #     cv.putText(next_gray, translation, (50,150), font, fontScale, fontColor, lineType)
        cv.putText(next_gray, str(dist1), (50, 90), font, fontScale, fontColor, lineType)
        cv.putText(next_gray, str(dist2), (50, 150), font, fontScale, fontColor, lineType)
        cv.putText(next_gray, str(dist3), (50, 210), font, fontScale, fontColor, lineType)
        #cv.putText(next_gray, motion_type, (50, 150), font, fontScale, fontColor, lineType)

        # write the frame to the new video

        if not os.path.exists(save_name[:-4]):
            os.makedirs(save_name[:-4])
        print(count)
        cv.imwrite(os.path.join(save_name[:-4], str(count) + ".jpg"), next_gray)

        outputStream.write(next_gray)

        # update the previous frame
        prvs = nxt.copy()
        count += 1

    # outputStream.release()

from img2vec_pytorch.img_to_vec import Img2Vec
from PIL import Image
def IMG2VEC(model,img):
    img = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)
    return model.get_vec(Image.fromarray(img))

def process_video_with_img2vec():
    # specify directory and file name
    dir_path = "/home/qilei/DEVELOPMENT/dataset_preprocess/temp_datas/Camera_motion_estimation/"
    filename = "20211026_1006_5004_c_9.22-12.29_crop.avi"

    # initialise stream from video
    cap = cv.VideoCapture(os.path.join(dir_path, filename))
    ret, prvs = cap.read()

    # initialise video writer
    frameRate = int(cap.get(cv.CAP_PROP_FPS))
    codec = cv.VideoWriter_fourcc(*'XVID')
    save_name = os.path.join(dir_path, "im2vec_motion_" + filename)
    outputStream = cv.VideoWriter(save_name, codec, frameRate, (int(cap.get(3)), int(cap.get(4))))

    # set parameters for text drawn on the frames
    font = cv.FONT_HERSHEY_COMPLEX
    fontScale = 2
    fontColor = (68, 148, 213)
    lineType = 3

    # initialise text variables to draw on frames
    angle = 'None'
    translation = 'None'
    motion = 'None'
    motion_type = 'None'
    dist1 = 0
    dist2 = 0
    dist3 = 0
    # set counter value
    count = 1
    img2vec = Img2Vec()
    # main loop
    while True:
        # read a new frame
        ret, nxt = cap.read()

        if not ret:
            break
        scale_percent = 50  # percent of original size
        width = int(nxt.shape[1] * scale_percent / 100)
        height = int(nxt.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        nxt = cv2.resize(nxt, dim, interpolation=cv2.INTER_AREA)

        # if the image is colored
        if len(prvs.shape) == 3:
            prvs_gray = cv.cvtColor(prvs.copy(), cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(nxt.copy(), cv.COLOR_BGR2GRAY)
        else:
            prvs_gray = prvs.copy()
            next_gray = nxt.copy()

        if count > 1:
            '''
            pre_fv_hu_moments = fd_hu_moments(prvs_gray)
            pre_fv_haralick = fd_haralick(prvs_gray)
            pre_fv_histogram = fd_histogram(prvs)

            next_fv_hu_moments = fd_hu_moments(next_gray)
            next_fv_haralick = fd_haralick(next_gray)
            next_fv_histogram = fd_histogram(nxt)

            dist1 = round(np.linalg.norm(pre_fv_hu_moments - next_fv_hu_moments),8)
            dist2 = round(np.linalg.norm(pre_fv_haralick - next_fv_haralick),8)
            dist3 = round(np.linalg.norm(pre_fv_histogram - next_fv_histogram),8)
            '''
            pre_f = IMG2VEC(img2vec,prvs)

            next_f = IMG2VEC(img2vec,nxt)

            motion = round(1-cosine_similarity(pre_f.reshape((1, -1)), next_f.reshape((1, -1)))[0][0],8)

            #motion_type = "mix"


        # put values on the frame
        #     cv.putText(next_gray, angle, (50,100), font, fontScale, fontColor, lineType)
        #     cv.putText(next_gray, translation, (50,150), font, fontScale, fontColor, lineType)
        cv.putText(nxt, str(motion), (50, 90), font, fontScale, fontColor, lineType)
        #cv.putText(next_gray, motion_type, (50, 150), font, fontScale, fontColor, lineType)

        # write the frame to the new video

        if not os.path.exists(save_name[:-4]):
            os.makedirs(save_name[:-4])
        print(count)
        cv.imwrite(os.path.join(save_name[:-4], str(count) + ".jpg"), nxt)

        outputStream.write(next_gray)

        # update the previous frame
        prvs = nxt.copy()
        count += 1

if __name__ == "__main__":
    #process_video_with_opt_flow()
    process_video_with_global_feature()
    #process_video_with_img2vec()