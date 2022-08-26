#批量修改尺寸
import imp
import os
from PIL import Image
from PIL import ImageFile
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True

foto_base_path = '/media/zky/T7/task3/RAFT/frame/L00_frame'
foto_names_path = os.listdir(foto_base_path)
foto_names = foto_names_path[0:]
foto_names = sorted(foto_names)
def foto_resize():
    for foto_name in foto_names:

        dir_origin_img="/media/zky/T7/task3/RAFT/frame/L00_frame/"+foto_name+"/"#path for read
        dir_resize_save="/media/zky/T7/task3/RAFT/frame_small/L00_frame/"+foto_name+"/"#path for save
        os.makedirs(dir_resize_save)
        size=(960,540)

        #获取目录下所有图片名
        list_temp = os.listdir(dir_origin_img)
        list_img = list_temp[0:] #从第一项开始取
        #list_img = sorted(list_img)

        #获得路径、打开要修改的图片
        for img_name in list_img:
            img_path = dir_origin_img+img_name
            old_image = Image.open(img_path)
            save_path = dir_resize_save+img_name

            #保存修改尺寸后的图片
            old_image.resize(size, Image.ANTIALIAS).save(save_path)
        print("Done!")
foto_resize()
