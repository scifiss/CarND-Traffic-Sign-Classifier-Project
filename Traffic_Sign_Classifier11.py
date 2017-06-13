# -*- coding: utf-8 -*-
"""
Based on Classifier 3, only add accuracy and loss plot
"""

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file='traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

train_coords = train['coords']
train_sizes = train['sizes']
valid_coords = valid['coords']
valid_sizes = valid['sizes']
test_coords = test['coords']
test_sizes = test['sizes']

import numpy as np
from PIL import Image
# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape= X_train[0].shape
Height,Width,Channel = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt

# Visualizations will be shown in the notebook.
#%matplotlib inline
import cv2
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    arr = arr.astype('uint8')
    return arr
# select an image
#index = random.randint(0,len(X_train))

#image = normalize(X_train[index].squeeze())
#
#plt.imshow(image)
#img = Image.fromarray(image,'RGB')
#enhancer = ImageEnhance.Contrast(img)
#img_contrast = enhancer.enhance(2.0)
#img1 = np.asarray(img_contrast)
#plt.imshow(img1)

#print(y_train[index])

import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 128



# crop image area within ROI, and enlarge it
for ipic in range( n_train):    
    img = normalize(X_train[ipic].squeeze())
    x1 = (train_coords[ipic][0]/train_sizes[ipic][0]*Height).astype('uint8')
    x2 = (train_coords[ipic][2]/train_sizes[ipic][0]*Height).astype('uint8')
    y1 = (train_coords[ipic][1]/train_sizes[ipic][1]*Width).astype('uint8')
    y2 = (train_coords[ipic][3]/train_sizes[ipic][1]*Width).astype('uint8')
    img = img[x1:x2,y1:y2]
    X_train[ipic] = cv2.resize(img,(Width,Height))
    
for ipic in range( n_validation):    
    img = normalize(X_valid[ipic].squeeze())
    x1 = (valid_coords[ipic][0]/valid_sizes[ipic][0]*Height).astype('uint8')
    x2 = (valid_coords[ipic][2]/valid_sizes[ipic][0]*Height).astype('uint8')
    y1 = (valid_coords[ipic][1]/valid_sizes[ipic][1]*Width).astype('uint8')
    y2 = (valid_coords[ipic][3]/valid_sizes[ipic][1]*Width).astype('uint8')
    img = img[x1:x2,y1:y2]
    X_valid[ipic] = cv2.resize(img,(Width,Height))

k=0    
for ipic in range( n_test):
    img = normalize(X_test[ipic].squeeze())
    if (test_sizes[ipic][0]>test_coords[ipic][2]):
        x1 = (test_coords[ipic][0]/test_sizes[ipic][0]*Height).astype('uint8')
        x2 = (test_coords[ipic][2]/test_sizes[ipic][0]*Height).astype('uint8')
        y1 = (test_coords[ipic][1]/test_sizes[ipic][1]*Width).astype('uint8')
        y2 = (test_coords[ipic][3]/test_sizes[ipic][1]*Width).astype('uint8')
        img = img[x1:x2,y1:y2]
        X_test[ipic] = cv2.resize(img,(Width,Height))
    else:
        X_test[ipic] = img
        k +=1
            
    
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    #print(x.shape)   
    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 12), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(12))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    #conv1 = tf.nn.dropout(conv1, keep_prob1)

    # Pooling. Input = 28x28x12. Output = 14x14x12    .
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
       
    # SOLUTION: Layer 2: Convolutional. 14x14x20  Output = 10x10x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    #conv2 = tf.nn.dropout(conv2, keep_prob2)

    # SOLUTION: Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x32. Output = 800.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 800. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 360), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(360))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob3)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(360, 120), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(120))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob4)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(120, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_prob4 = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

accValid=[]
accTrain=[]
lossValid=[]
lossTrain=[]

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss,accuracy = sess.run([loss_operation,accuracy_operation], feed_dict={x: batch_x, y: batch_y,keep_prob3:1,keep_prob4:1})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss/num_examples

from sklearn.utils import shuffle
goodpoint = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        
        training_accuracy = 0
        training_loss = 0        
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob3:0.5,keep_prob4:0.5})
            loss,accuracy = sess.run([loss_operation,accuracy_operation], feed_dict={x: batch_x, y: batch_y,keep_prob3:0.5,keep_prob4:0.5})
            training_accuracy += (accuracy * len(batch_x))
            training_loss += (loss * len(batch_x))
        training_accuracy = training_accuracy/len(X_train) 
        training_loss = training_loss/len(X_train)
            
        validation_accuracy, validation_loss = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
       # if i % 2 is 0:
        accTrain.append(training_accuracy)
        lossTrain.append(training_loss)
        accValid.append(validation_accuracy)
        lossValid.append(validation_loss)
        
        if validation_accuracy>0.95:
            goodpoint +=1
            if goodpoint>10:
                break
        
    saver.save(sess, './lenet.chkp')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy, test_loss = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))



fig, ax = plt.subplots()

ax.plot(range(0,len(lossValid)),accTrain, label='Training')
ax.plot(range(0,len(lossValid)),accValid, label='Validation')
ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.2,1.2])
ax.legend(loc=4)

fig, ax = plt.subplots()

ax.plot(range(0,len(lossValid)),lossTrain, label='Training')
ax.plot(range(0,len(lossValid)),lossValid, label='Validation')
ax.set_xlabel('Training steps')
ax.set_ylabel('Loss')
ax.set_ylim([0,1.2])
ax.legend(loc=1)
plt.show()




