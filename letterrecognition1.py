import numpy as np
import os
from scipy import ndimage
import pickle

image_size=28
def load_letters(folder):
    image_files=os.listdir(folder)
    dataset=np.ndarray(shape=(len(image_files),image_size,image_size),dtype=np.float32)
    i=0

    for image in image_files:
        try:
            imageaddress=os.path.join(folder,image)
            a=(ndimage.imread(imageaddress).astype(float)-128.0)/255.0
            dataset[i,:,:]=a
            i=i+1
        except:
            pass
    dataset=dataset[0:i,:,:]
    return dataset

def maybe_pickle(data_folders):
    dataset_names=[]
    for folder in data_folders:
        set_filename=folder+'.pickle'
        dataset_names.append(set_filename)

        dataset=load_letters(folder)

        with open(set_filename,'wb') as f:
            pickle.dump(dataset,f)
    return dataset_names

trainfolders=['A','B','C','D','E','F','G','H','I','J']
train_datasets=maybe_pickle(trainfolders)
