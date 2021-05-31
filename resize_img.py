import os, shutil
import cv2



def resize_img (src_database, target_database) :
    
    if ( os.path.exists(src_database)) :

        if os.path.exists(target_database):    
            shutil.rmtree(target_database)
            os.mkdir(target_database)

        for each in os.listdir(src_database):
            img = cv2.imread(os.path.join(src_database,each))
            img = cv2.resize(img,(256,256))
            cv2.imwrite(os.path.join(target_database,each), img) 