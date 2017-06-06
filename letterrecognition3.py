import tensorflow as tf
import pickle
import numpy as np

picklefile1='notMNISTtest.pickle'
picklefile2='notMNISTtrain.pickle'

with open(picklefile1,'rb') as f1:
    dictionary=pickle.load(f1)
    test_dataset=dictionary['test_dataset']
    test_labels=dictionary['test_labels']

with open(picklefile2,'rb') as f2:
    dict=pickle.load(f2)
    train_dataset=dict['train_dataset']
    train_labels=dict['train_labels']
    valid_dataset=dict['valid_dataset']
    valid_labels=dict['valid_labels']

def vectorise(label):
    a=np.zeros((1,10),dtype=np.float32)
    a[0,label]=1.0
    return a


def reformat(dataset,labels):
    dataset=dataset.reshape((-1,28*28)).astype(np.float32)
    labelnew=np.zeros((len(labels),10),dtype=np.float32)
    count=0
    for label in labels:
        labelnew[count,:]=vectorise(label)
        count+=1
    return dataset,labelnew

train_dataset,train_labels=reformat(train_dataset,train_labels)
test_dataset,test_labels=reformat(test_dataset,test_labels)
valid_dataset,valid_labels=reformat(valid_dataset,valid_labels)


train_subset=10000

graph=tf.Graph()
with graph.as_default():
    tf_train_dataset=tf.constant(train_dataset[:train_subset,:])
    tf_train_labels=tf.constant(train_labels[:train_subset,:])
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)

    weights=tf.Variable(tf.truncated_normal([28*28,10]))
    baises=tf.Variable(tf.zeros([10]))

    logits=tf.matmul(tf_train_dataset,weights)+baises
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))

    optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    train_prediction=tf.nn.softmax(tf.matmul(train_dataset,weights)+baises)
    valid_prediction=tf.nn.softmax(tf.matmul(valid_dataset,weights)+baises)
    test_prediction=tf.nn.softmax(tf.matmul(test_dataset,weights)+baises)

steps=801;
def accuracy(prediction,lab):
     return (100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(lab, 1))
          / prediction.shape[0])

with tf.Session(graph=graph) as session:
    session.run(tf.initialize_all_variables())
    for step in range(steps):
        _,l,t,v=session.run([optimizer,loss,train_prediction,valid_prediction])
	if(step%50==0):
		print ('loss:',l,' training accuracy:',accuracy(t,train_labels),' validation accuracy:',accuracy(v,valid_labels))

    print('Training accuracy: %lf' %accuracy(train_prediction.eval(),train_labels))
    print('Validation accuracy: %lf' %accuracy(valid_prediction.eval(),valid_labels))
    print('Testing accuracy: %lf' %accuracy(test_prediction.eval(),test_labels))
