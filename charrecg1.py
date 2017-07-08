import os
import numpy as np
import scipy
from scipy import ndimage
import pickle


folders=["%d"%(i) for i in range(1,63)]
train_dataset=np.ndarray((200*62,1600),dtype=np.float32)
train_labels=np.ndarray((200*62,1),dtype=np.float32)
valid_dataset=np.ndarray((50*62,1600),dtype=np.float32)
valid_labels=np.ndarray((50*62,1),dtype=np.float32)
test_dataset=np.ndarray((40*62,1600),dtype=np.float32)
test_labels=np.ndarray((40*62,1),dtype=np.float32)
t=0
v=0
te=0
j=-1
k=0
for folder in folders:
    print folder
    images=os.listdir(folder)
    images.sort()
    j+=1
    k=0
    for image in images:
        resizedi=np.ndarray((40,40),dtype=np.float32)
        imagepath=os.path.join(folder,image)
        img=ndimage.imread(imagepath,flatten=1).astype(float)
        resizedi=scipy.misc.imresize(img,(40,40))
        resizedi=np.reshape(resizedi,(1,1600))
	resizedi=resizedi/255
        if(k<200):
	    train_dataset[t,:]=resizedi
            train_labels[t,:]=j
            t+=1
	elif(k>=200 and k<250):
	    valid_dataset[v,:]=resizedi
            valid_labels[v,:]=j
            v+=1
	elif(k>=250 and k<290):
	    test_dataset[te,:]=resizedi
            test_labels[te,:]=j
            te+=1
	else:
	    break
	k+=1	
	
print train_dataset.shape,train_labels.shape
print valid_dataset.shape,valid_labels.shape
print test_dataset.shape,test_labels.shape

dict={}
dict['train_dataset']=train_dataset
dict['train_labels']=train_labels
dict['valid_dataset']=valid_dataset
dict['valid_labels']=valid_labels
dict['test_dataset']=test_dataset
dict['test_labels']=test_labels
with open('train_dataset.pickle','wb') as f:
    pickle.dump(dict,f)
