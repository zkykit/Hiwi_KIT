import cv2
import os
import sys
#from pandas import to_datetime



video_base_path = '/media/zky/T7/task3/RAFT/video'
videos = os.listdir(video_base_path)
videos = sorted(videos)
i=0

video_name = videos[i]

print(video_name)
#i=i+1
video_path = '/media/zky/T7/task3/RAFT/frame/'+video_name+'_frame'#folder name for pictures
video_LXX_path = '/media/zky/T7/task3/RAFT/video/'+video_name
#print(video_LXX_path) /media/zky/T7/task3/RAFT/video/L00

videos = os.listdir(video_LXX_path)
for video_name in videos:
    file_name = video_name.split('.')[0]#file_name is '2022_03_10_S05_SC01_T01_SE003_L00'
    folder_name = video_path +'/'+file_name#folder_name is 'L00_frame2022_03_10_S05_SC01_T01_SE003_L00' 改这行
    os.makedirs(folder_name, exist_ok=True)#creat folder for frames
    vc = cv2.VideoCapture(video_LXX_path+'/'+video_name) 
    c=0
    rval=vc.isOpened()

    while rval:  
        c = c + 1
        rval, frame = vc.read()
        pic_path = folder_name+'/'#图片的路径
        if rval:
            cv2.imwrite(pic_path + str(c) + '.png', frame)
            cv2.waitKey(1)
        else:
            break
vc.release()
print('save_success')
print(folder_name)


