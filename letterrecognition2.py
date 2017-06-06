import numpy as np
import os
from scipy import ndimage
import pickle

def merge_datasets(picklefiles,trainsize,validasize):
    letterset=np.ndarray((30000,28,28),dtype=np.float32)	
    validdata=np.ndarray((validasize,28,28),dtype=np.float32)
    traindata=np.ndarray((trainsize,28,28),dtype=np.float32)
    validlabel=np.ndarray((validasize),dtype=np.int32)
    trainlabel=np.ndarray((trainsize),dtype=np.int32)
    validasize_perclass=validasize/10
    trainsize_perclass=trainsize/10
    startt=0
    startv=0
    endv=validasize_perclass
    endt=trainsize_perclass
    endl=trainsize_perclass+validasize_perclass
    for label,picklee in enumerate(picklefiles):
        with open(picklee,'rb') as f:
                letterset=pickle.load(f)
                validdata[startv:endv,:,:]=letterset[0:validasize_perclass,:,:]
                validlabel[startv:endv]=label

                traindata[startt:endt,:,:]=letterset[validasize_perclass:endl,:,:]
                trainlabel[startt:endt]=label
                startv+=validasize_perclass;
                endv+=validasize_perclass;
                startt+=trainsize_perclass
                endt+=trainsize_perclass
    return validdata,validlabel,traindata,trainlabel

train_size=200000
valid_size=10000
picklefiles=['A.pickle','B.pickle','C.pickle','D.pickle','E.pickle','F.pickle','G.pickle','H.pickle','I.pickle','J.pickle']
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  picklefiles, train_size, valid_size)



def shuffledata(dataset,labels):
    per=np.random.permutation(labels.shape[0])
    dataset=dataset[per,:,:]
    labels=labels[per]
    return dataset, labels

train_dataset,train_labels=shuffledata(train_dataset,train_labels)
valid_dataset,valid_labels=shuffledata(valid_dataset,valid_labels)


f=open('notMNISTtrain.pickle','wb')
savediction={
'train_dataset':train_dataset,
'train_labels':train_labels,
'valid_dataset':valid_dataset,
'valid_labels':valid_labels,
}

pickle.dump(savediction,f)
