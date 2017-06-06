import numpy as np
import os
from scipy import ndimage
import pickle


def merge_datasets(picklefiles,testsize):	
    letterset=np.ndarray((20000,28,28),dtype=np.float32)	
    testdata=np.ndarray((testsize,28,28),dtype=np.float32)
    testlabel=np.ndarray((testsize),dtype=np.int32)
    testsize_perclass=testsize/10
    startt=0
    endt=testsize_perclass
    for label,picklee in enumerate(picklefiles):
        with open(picklee,'rb') as f:
		letterset=pickle.load(f)
        	testdata[startt:endt,:,:]=letterset[0:testsize_perclass,:,:]
                testlabel[startt:endt]=label
                startt+=testsize_perclass
                endt+=testsize_perclass


    return testdata,testlabel

test_size=10000
picklefiles=['A.pickle','B.pickle','C.pickle','D.pickle','E.pickle','F.pickle','G.pickle','H.pickle','I.pickle','J.pickle']
test_dataset, test_labels = merge_datasets(picklefiles, test_size)


def shuffledata(dataset,labels):
    per=np.random.permutation(labels.shape[0])
    dataset=dataset[per,:,:]
    labels=labels[per]
    return dataset, labels

test_dataset,test_labels=shuffledata(test_dataset,test_labels)


f=open('notMNISTtest.pickle','wb')
savediction={
'test_dataset':test_dataset,
'test_labels':test_labels,
}

pickle.dump(savediction,f)

