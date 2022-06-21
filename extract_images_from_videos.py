import cv2
import glob
import os
def images_extraction():
    video_list = glob.glob("/home/qilei/DATASETS/trans_drone/trans_drone_videos2/*.MOV")

    steps=240

    offset = 120

    for video_dir in video_list:

        cap = cv2.VideoCapture(video_dir)

        frame_index = 1

        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)

        success,frame = cap.read()

        

        video_name = os.path.basename(video_dir)

        video_name = video_name.replace(".MOV","")

        

        while success:

            

            if (frame_index-offset)%steps==0:

                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)

                success,frame = cap.read()

                if success:

                    img_name = video_name+"_"+str(frame_index)+".jpg"

                    cv2.imwrite("/home/qilei/DATASETS/trans_drone/trans_drone_videos2_images2/"+img_name,frame)



            frame_index+=1

images_extraction()
