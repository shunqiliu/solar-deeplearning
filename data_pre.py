import os
import numpy as np
import cv2
from tqdm import tqdm

def load_origin_pic_label_feat(file_dir='..\Solar_Panel_Soiling_Image_dataset\PanelImages'):   
    L=[]  
    R=[]   
    I=[]   
    print('Load data')
    for root, dirs, files in os.walk(file_dir):  
        for file in tqdm(files):  
            if os.path.splitext(file)[1] == '.jpg': 
                string=os.path.join( file).replace('.jpg','').split('_')
                string=[float(string[4]),float(string[6]),float(string[8]),float(string[-1]),float(string[-3])]
                L.append(string[-1])
                R.append(string[:4])
                F = cv2.imread(os.path.join(root, file))
                I.append(F)
    print('Convert data')
    L=np.array(L)
    R=np.array(R)
    I=np.array(I)
    print('Save data')
    np.save('data_npy/label.npy',L )
    np.save('data_npy/feats.npy',R ) 
    np.save('data_npy/image.npy',I )