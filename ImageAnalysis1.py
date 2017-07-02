#non text filter,attached characters in binary image, a appearing 3
import scipy
from scipy import ndimage
import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter
import copy
import os
import pickle
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
import random


def char(j):
    if(j<=9):
	return '%d'%(j)
    if(j==10):
	return 'A'
    if(j==11):
	return 'B'
    if(j==12):
	return 'C'
    if(j==13):
	return 'D'
    if(j==14):
	return 'E'
    if(j==15):
	return 'F'
    if(j==16):
	return 'G'
    if(j==17):
	return 'H'
    if(j==18):
	return 'I'
    if(j==19):
	return 'J'
    if(j==20):
	return 'K'
    if(j==21):
	return 'L'
    if(j==22):
	return 'M'
    if(j==23):
	return 'N'
    if(j==24):
	return 'O'
    if(j==25):
	return 'P'
    if(j==26):
	return 'Q'
    if(j==27):
	return 'R'
    if(j==28):
	return 'S'
    if(j==29):
	return 'T'
    if(j==30):
	return 'U'
    if(j==31):
	return 'V'
    if(j==32):
	return 'W'
    if(j==33):
	return 'X'
    if(j==34):
	return 'Y'
    if(j==35):
	return 'Z'
    if(j==36):
	return 'a'
    if(j==37):
	return 'b'
    if(j==38):
	return 'c'
    if(j==39):
	return 'd'
    if(j==40):
	return 'e'
    if(j==41):
	return 'f'
    if(j==42):
	return 'g'
    if(j==43):
	return 'h'
    if(j==44):
	return 'i'
    if(j==45):
	return 'j'
    if(j==46):
	return 'k'
    if(j==47):
	return 'l'
    if(j==48):
	return 'm'
    if(j==49):
	return 'n'
    if(j==50):
	return 'o'
    if(j==51):
	return 'p'
    if(j==52):
	return 'q'
    if(j==53):
	return 'r'
    if(j==54):
	return 's'
    if(j==55):
	return 't'
    if(j==56):
	return 'u'
    if(j==57):
	return 'v'
    if(j==58):
	return 'w'
    if(j==59):
	return 'x'
    if(j==60):
	return 'y'
    if(j==61):
	return 'z'
	
	
seed = 128
rng = np.random.RandomState(seed)
input_num_units = 1600
hidden_num_units = 100
output_num_units = 62

# create model
model = Sequential([
  Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu',name='charrecg1'),
  Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax',name='charrecg2'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('charrecgweights.h5')

def identifychar(acharimg):
    resizedi=np.ndarray((40,40),dtype=np.float32)
    reshapedi=np.ndarray((1,1600),dtype=np.float32)			
    resizedi=copy.deepcopy(acharimg)
    reshapedi=np.reshape(resizedi,(1,1600))
    reshapedi=reshapedi/255
    return char(model.predict_classes(reshapedi))
	

def seperatelines(region1,region2,newimg):
    top_left1=np.min(region1, axis=0)
    bottom_right1 = np.max(region1, axis=0)
    top_left2=np.min(region2, axis=0)
    bottom_right2 = np.max(region2, axis=0)

    h1=abs(bottom_right1[1]-top_left1[1])
    h2=abs(bottom_right2[1]-top_left2[1])
    h=min(h1,h2)
    w1=abs(top_left1[0]-bottom_right1[0])
    w2=abs(top_left2[0]-bottom_right2[0])

    center1=[(top_left1[1]+bottom_right1[1])/2,(top_left1[0]+bottom_right1[0])/2]
    center2=[(top_left2[1]+bottom_right2[1])/2,(top_left2[0]+bottom_right2[0])/2]
    dx=abs(center1[1]-center2[1])-((w1+w2)/2)
    dy=abs(center1[0]-center2[0])
    dyy=abs(center1[0]-center2[0])-((h1+h2)/2)
    if(dyy<=0 and dx<=0):
        if((center1[0]<center2[0]) and ((center1[0]<top_left2[1]) and (center2[0]>bottom_right1[1]))):
            print 'yes'
            for j in range(top_left2[0],bottom_right2[0]+1):
                for i in range(top_left2[1],bottom_right1[1]+2):
                    newimg[i,j]=0
        elif((center1[0]>center2[0]) and ((center1[0]>bottom_right2[1]) and (center2[0]<top_left1[1]))):
            print 'yess'
            for j in range(top_left1[0],bottom_right1[0]+1):
                for i in range(top_left1[1],bottom_right2[1]+2):
                    newimg[i,j]=0


def groupcharacters(region1,region2,newimg):
    top_left1=np.min(region1, axis=0)
    bottom_right1 = np.max(region1, axis=0)
    top_left2=np.min(region2, axis=0)
    bottom_right2 = np.max(region2, axis=0)

    h1=abs(bottom_right1[1]-top_left1[1])
    h2=abs(bottom_right2[1]-top_left2[1])
    h=min(h1,h2)
    w1=abs(top_left1[0]-bottom_right1[0])
    w2=abs(top_left2[0]-bottom_right2[0])

    center1=[(top_left1[1]+bottom_right1[1])/2,(top_left1[0]+bottom_right1[0])/2]
    center2=[(top_left2[1]+bottom_right2[1])/2,(top_left2[0]+bottom_right2[0])/2]
    dx=abs(center1[1]-center2[1])-((w1+w2)/2)
    dy=abs(center1[0]-center2[0])
    k=0.43
    #if(w1/h1<0.6  and w2/h2<0.6):
    #    k=0.36
    if((dx<=k*h) and (dy<=0.6*h) and (abs(h1-h2)<=1.2*h)):
        if((center1[1]<center2[1]) and (bottom_right1[0]<top_left2[0])):  #nonoverlapping
            for i in range(top_left1[1],bottom_right1[1]+1):
                for j in range(bottom_right1[0],top_left2[0]+1):
                    newimg[i,j]=255
        elif((center2[1]<center1[1]) and (bottom_right2[0]<top_left1[0])):  #nonoverlapping
            for i in range(top_left2[1],bottom_right2[1]+1):
                for j in range(bottom_right2[0],top_left1[0]+1):
                    newimg[i,j]=255




def extract_region(img, region):
    top_left=np.min(region, axis=0)
    bottom_right = np.max(region, axis=0)
    margin = 2
    region_of_image = img.copy()[top_left[1]-margin:bottom_right[1]+margin, top_left[0]-margin:bottom_right[0]+margin]
    return region_of_image

def marktext(newimg,bb):
    top_left=[bb[0,1],bb[0,0]]
    bottom_right=[bb[1,1],bb[1,0]]
    for i in range(int(top_left[1]),int((bottom_right[1]))+1):
        for j in range(int(top_left[0]),int((bottom_right[0]))+1):
            newimg[j,i]=255.0
    return newimg

def bbox (points):
    res = np.zeros((2,2))
    res[0,:] = np.min(points, axis=0)
    res[1,:] = np.max(points, axis=0)
    return res

def bbox_width(bbox):
    return (bbox[1,0] - bbox[0,0] + 1)

def bbox_height(bbox):
    return (bbox[1,1] - bbox[0,1] + 1)

def invertimage(thesh2):
    for i in range(0,thesh2.shape[0]):
        for j in range(0,thesh2.shape[1]):
            thesh2[i,j]=255-thesh2[i,j]
    return thesh2

def checkoverlapping(j,chars):
    x,y,w,h=chars[j]
    for i in range(0,len(chars)):
	if(i==j):
	    continue
	x1,y1,w1,h1=chars[i]
	if((x1<=x) and (x1+w1>=x+w) and (y1<=y) and (y1+h1>=y+h)):
	    return 1
    return 0

def sortwords(words):
    i=1
    while(i<len(words)):
        for j in range(0,i):
            x1,y1,w1,h1=words[i]
            x2,y2,w2,h2=words[j]
            if(x1<x2 and y2+h2>y1):
                t=words[i]
                words[i]=words[j]
                words[j]=t
                i-=1
                break
        i+=1
    return words
def checkotherpart(chars,j):
    x1,y1,w1,h1=chars[j]
    for i in range(0,len(chars)):
	if(i==j):
	    continue
	x2,y2,w2,h2=chars[i]
	center1=[1.0*(y1+y1+h1)/2,1.0*(x1+x1+w1)/2]
	center2=[1.0*(y2+y2+h2)/2,1.0*(x2+x2+w2)/2]
	if((center1[1]>x2 and center1[1]<x2+w2) or (center2[1]>x1 and center2[1]<x1+w1)):
	    return i
    return -1	
    

mser = cv2.MSER_create()
img = cv2.imread('A.jpg')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayimg.jpg',grayimg)
regions,_ = mser.detectRegions(grayimg)                  #height,width
newimg=np.ndarray((grayimg.shape[0],grayimg.shape[1]),dtype=np.float32)
newimg[:,:]=0
i=0

while(i<len(regions)):
	
    bb = bbox(regions[i])
    w=bbox_width(bb)
    h=bbox_height(bb)	
    top_left=[bb[0,1],bb[0,0]]
    bottom_right=[bb[1,1],bb[1,0]]
    dx=abs(top_left[0]-bottom_right[0])#width
    dy=abs(top_left[1]-bottom_right[1])#height
    if(dy==0):
	regions.pop(i)
	continue
    ratio=dx*1.00/dy
    
    if((ratio<0.05) or (ratio>10)):
	    regions.pop(i)
	    continue
    if((1.00*dy/grayimg.shape[0]>0.8) or (1.00*dy/grayimg.shape[0]<0.001)):
	    regions.pop(i)
	    continue
    if((1.00*dx/grayimg.shape[1]>0.8) or (1.00*dx/grayimg.shape[1]<0.0001)):
	    regions.pop(i)
	    continue
    
    #cv2.imwrite('%d.png'%(i),extract_region(img, regions[i]))
    newimg=marktext(newimg,bb)
    i+=1

cv2.imwrite('newimg.png',newimg)

for i in range(0,len(regions)):
    for j in range(i+1,len(regions)):
        groupcharacters(regions[i],regions[j],newimg)
'''
for i in range(0,len(regions)):
    for j in range(i+1,len(regions)):
        seperatelines(regions[i],regions[j],newimg)
'''
cv2.imwrite('newimg1.jpg',newimg)

for i in range(0,newimg.shape[0]):
    for j in range(0,newimg.shape[1]):
        try:
            if((newimg[i,j-1]==255 and newimg[i-1,j]==255) or (newimg[i-1,j]==255 and newimg[i,j+1]==255) or(newimg[i,j+1]==255 and newimg[i+1,j]==255) or (newimg[i+1,j]==255 and newimg[i,j-1]==255)):
                newimg[i,j]=255
        except:
            pass
i=newimg.shape[0]-1
while(i>=0):
    for j in range(0,newimg.shape[1]):
        try:
            if((newimg[i,j-1]==255 and newimg[i-1,j]==255) or (newimg[i-1,j]==255 and newimg[i,j+1]==255) or(newimg[i,j+1]==255 and newimg[i+1,j]==255) or (newimg[i+1,j]==255 and newimg[i,j-1]==255)):
                newimg[i,j]=255
        except:
            pass
    i-=1
for i in range(0,newimg.shape[0]):
    j=newimg.shape[1]-1
    while(j>=0):
        try:
            if((newimg[i,j-1]==255 and newimg[i-1,j]==255) or (newimg[i-1,j]==255 and newimg[i,j+1]==255) or(newimg[i,j+1]==255 and newimg[i+1,j]==255) or (newimg[i+1,j]==255 and newimg[i,j-1]==255)):
                newimg[i,j]=255
        except:
            pass
        j-=1
i=newimg.shape[0]-1
while(i>=0):
    j=newimg.shape[1]-1
    while(j>=0):
        try:
            if((newimg[i,j-1]==255 and newimg[i-1,j]==255) or (newimg[i-1,j]==255 and newimg[i,j+1]==255) or(newimg[i,j+1]==255 and newimg[i+1,j]==255) or (newimg[i+1,j]==255 and newimg[i,j-1]==255)):
                newimg[i,j]=255
        except:
            pass
        j-=1
    i-=1

cv2.imwrite('newimg2.jpg',newimg)
words=[]
nimg = cv2.imread('newimg2.jpg',0)
ret,thresh = cv2.threshold(nimg,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh, 1, 2)

for i in range(0,len(contours)):
    cnt = contours[i]
    words.append(cv2.boundingRect(cnt))
words.sort(key=itemgetter(1)) #words[i]=x,y,w,h
words=sortwords(words)
jj=0
for i in range(0,len(words)):
    x,y,w,h=words[i]
    wordimg=grayimg.copy()[y-1:y+h+2, x-1:x+w+2]

    blur = cv2.GaussianBlur(wordimg,(5,5),0)
    ret2,thresh2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    count1=0
    count2=0
    for i in range(0,h+3):
	if(thresh2[i,0]==0):
	    count1+=1
	if(thresh2[i,w+2]==0):
	    count1+=1
	if(thresh2[i,0]==255):
	    count2+=1
	if(thresh2[i,w+2]==255):
	    count2+=1
    for i in range(0,w+3):
	if(thresh2[0,i]==0):
	    count1+=1
	if(thresh2[h+2,i]==0):
	    count1+=1
	if(thresh2[0,i]==255):
	    count2+=1
	if(thresh2[h+2,i]==255):
	    count2+=1
	
    if(count2>count1):
        thresh2=invertimage(thresh2)
    #cv2.imwrite('w%d.jpg'%(i),thresh2)
    ret3,thresh3 = cv2.threshold(thresh2,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh3, 1, 2)
					
    wrd=''
    chars=[]
    for i in range(0,len(contours)):
        cnt = contours[i]
        chars.append(cv2.boundingRect(cnt))
    chars.sort(key=itemgetter(0))
    
    k=0	
    while(k<len(chars)):
        x1,y1,w1,h1=chars[k]
        if(checkoverlapping(k,chars)==1):
	    chars.pop(k)
	    continue
        k+=1	

    j=0
    
    while(j<len(chars)):
        x1,y1,w1,h1=chars[j]
        
	k=checkotherpart(chars,j)	
	if(k!=-1):
	    x2,y2,w2,h2=chars[k]		
	    if(y1<y2):
		x=min(x1,x2)
		w=max(x1+w1,x2+w2)
		h1=y2+h2-y1
		w1=w-x
		x1=x
		
	    elif(y2<y1):
		x=min(x1,x2)
		w=max(x1+w1,x2+w2)
		h1=y1+h1-y2
		w1=w-x
		x1=x
		y1=y2
	    chars.pop(k)
			
        charimg=thresh2.copy()[y1:y1+h1, x1:x1+w1]
        j+=1
        if(h1>=w1):
            a=30
            ratio=w1/(h1*1.0)
            b=(int)(30*ratio)
        else:
            b=30
            ratio=h1/(w1*1.0)
            a=(int)(ratio*30)
        rcharimg=scipy.misc.imresize(charimg,(a,b))
        rcharimg=invertimage(rcharimg)
        acharimg=np.zeros((40,40))
        acharimg[:,:]=255.0
        acharimg[20-(int)(a*1.0/2):20-(int)(a*1.0/2)+a,20-(int)(b*1.0/2):20-(int)(b*1.0/2)+b]=rcharimg
	cv2.imwrite('c%d.jpg'%(jj),acharimg)
	jj+=1
	wrd=wrd+identifychar(acharimg)
    print wrd	
        
