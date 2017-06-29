# **Traffic Sign Recognition** 

## Rebecca Gao

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And here is a link to my [project code](https://github.com/scifiss/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_rebecca.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculate summary statistics of the traffic signs data set:

![alt text](/examples/datasetall.png)
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43
Definition of all classes can be found at signnames.csv

#### 2. Include an exploratory visualization of the dataset.

For a small set of randomly selected classes, 3 sample images are randomly selected to show how they vary in brightness, size, color, and shapes.

![alt text](/examples/dataset_compare.png)

A bar chart shows numbers of images belonging to each class in the training, validation, and test dataset.

![alt text](/examples/datasethist.png)


### Design and Test a Model Architecture

#### 1) Describe how you preprocessed the image data.
What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

(1) Normalize the image

I decided to normalize the images because for each class, the included images vary a lot in the brightness. 
Two approaches are attempted to normalize images:
  (a) The image is converted to grayscale or intensity from (i.e. YUV, HSV, YCrCb), being equalized, and then converted back to RGB. Except YUV, the other colorspaces (i.e. HSV, YCrCb) resulted in weird color tones. For YUV normalization, the resulted RGB shows many dark patterns from the original light noises.
  (b) The image is normalized per color channel, so information from each color is honored. The processed images look better. I hereinafter normalize with second approach. 
In the plot below, first column is raw data, 2nd column is normalized in YUV space, 3rd column is normalized for each color.

![alt text](/examples/normalizing.png)

(2) Enhance the image

After normalization for individual channels, edge enhancing filters of different sizes are attempted since many images are blurred. In the plot below, first column is raw data, 2nd column is normalized data, 3rd column is applied with size=3 filter, and last column is applied with size=5 filter.
![alt text](/examples/edgeEnhancing.png)
Seemingly, edge detector does improve the image quality, and the one with size = 5 produces clearer images, but will also emphasize background edges and introduce artificial effect, like the darker edges around the arrow in the last image,and the middle in the 5th image. I am not sure if these effects will harm the learning process.
In the part of results and discussion, several runs are carried to see their effects on class recogniztion. It is seen that although an edge enhancing of size=5 improve the image visually, it deteriorates the classification. Edge enhancing with size=3 doesn't remarkably improve the classification. So in the final run, only normalization is used in preprocessing.

(3) Obtain ROI

I noticed in the German traffic sign website (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Annotationformat), the coordinates of region of interest (ROI) for the sign are offered. The coordinates are also included in our dataset. I would take advantage of the information and cut ROI out for recognition.
![alt text](/examples/Roi.png)

(4) Image augmentation

I decided to generate additional data because from the histogram we know the training dataset is very unbalanced. For those "minor" classes with little images, the probability of judging a new image as minor classes is small from learning. The objective is increasing minor class data so in the training set, all classes have almost equal data.
skimage.transform is used to combine all kinds of random transformation into one step. Thus the speed should be increased than consecutive image transformation.
Suplementrary images are added to original train dataset. For each "minor" class, i.e. class with less images, original images in the training dataset are selected randomly, underthrough scaling, rotation, shearing and translation to generate complementary images up to the size of 0.9 of the class with maximum sample size. 
Here is an example of an original image and augmented images:

![alt text](/examples/dataset_augmented.png)

The final result, shown on another file Traffic_Sign_Classifier_rebecca_withExtraImages.ipynb, is terrible. Therefore it is abandoned in my main workflow.

![alt text](/examples/resultExtraimage.png)

#### 2) Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   				|
| Flatten               | outputs 800                                   |
| Fully connected		| outptus 360  									|
| RELU					|												|
| dropout    	      	| with input variable keep_prob3 				|
| Fully connected		| outptus 120  									|
| RELU					|												|
| dropout    	      	| with input variable keep_prob4 				|
| Fully connected		| outptus 43  									|
The validation accuracy vs training accuracy, as well as the loss are shown below:

![alt text](/examples/finalAccuracy.png)
![alt text](/examples/finalLoss.png)

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used tf.nn.softmax_cross_entropy_with_logits to compute the distance from prediction to the labels, which is the loss that determines the convergence of the optimization process.
I use tf.train.AdamOptimizer to train the model with learning rate = 0.001. It uses Kingma and Ba's Adam algorithm, the moving averages of the parameters (momentum) to control the learning rate. A larger effective step size and scaled gradient offer better confergence than tf.train.GradientDescentOptimizer.
The batch size for moving average is 128 images, and a maximum of 100 epochs to limit the iteration times.
To ensure a good result, a consecutive succession of 10 epochs with validation > 0.95 is required to arrive at the stop.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.97
* validation set accuracy of 0.957
* test set accuracy of 0.944

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I have tried many options to improve the accuracy, most of them are failures.
1. With original LeNet provided in the project, with no normalization, the validation accuracy is already high, but from Epoch 10, the train set accuracy stays around 0.9 till Epoch 50. As more epochs are run, no considerable increase of the accuracy is observed. The validation set accuracy is 0.893.

![alt text](/examples/rawresult.png)

The inefficient learning means the learning graph needs to be improved, towards more nodes, more layers, and dropout, etc.

2. Different image enhancement techniques were tried in the preprocessing step.
Different methods of normalization, edge enhancing, and denoise are tried. In the plot below, 6 groups of images are shown, each with raw image, normalized in YUV space, and normalized in YUV space then denoised. For denoise, I use rof.denoise function.
The denoised images have more contrast (eg. 1,5,6), but there are still noises that blur the signs (eg.1). There are also some bad examples of denoising (eg 2,4).
![alt text](/examples/YuvDenoise.png)
Most denoiser or edge enhancement filter won’t work for the images here, because the sign has strokes of 1 or 2 pixels, too small to be recognized as information rather than noises. From the experiments, it is good to do some general preprocessing, but not of risky denoising/edge detector that is sensible to image texture and characteristic lengths.

A comparison of edge enhancing with the original LeNet are tried. Four experiments (normalization for each color, and then one with edge enhancing filter size=5, and two with size=3) are shown. Edge filter with size = 5 decreases the accuracy, and edge filter with size=3 have similar results with no edge filtering. 
![alt text](/examples/sharper3and5.png)

3. Find region of interest
It's noticed in the German traffic sign website (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Annotationformat) that the coordinates of region of interest (ROI) for the sign are offered. The coordinates are also included in our dataset. With the given LeNet, several runs of 100 epochs are tried and the test accuracy is above 0.9. I would take advantage of the information and cut ROI out for recognition.

4. Increase Neuron numbers (convolution neurons)
(1) first trial:

Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x12.
Pooling. Input = 28x28x12. Output = 14x14x12    .
Layer 2: Convolutional. Output = 10x10x32.
Pooling. Input = 10x10x32. Output = 5x5x32.
Layer 3: Fully Connected. Input = 800. Output = 120.
Layer 4: Fully Connected. Input = 120. Output = 84
Layer 5: Fully Connected. Input = 84. Output = 43.

After 29 epochs, test accuracy=0.929 with validation accuracy=0.951
I also tried other structures

Layer 1: Convolutional. Input = 32x32x3. Output = 32*32*32
Pooling. Input = 32*32*32. Output = 16*16*32   
Layer 2: Convolutional. 16*16*32  Output = 16*16*64.
Pooling. Input = 16*16*64.. Output = 8*8*64.
Layer 3: Input: 8*8*64  Output: 8*8*128
Pooling. Input = 8*8*128  Output = 8*8*128.
Flatten. Input = 8*8*128. Output = 8192.
Fully connected: 8192 -> 2048 -> 512 -> 43

The results are terrible. Validation accuracy hangs around 0.07 to 0.17.

5. Increase number of fully connected layer

Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x12.
Pooling. Input = 28x28x12. Output = 14x14x12    .
Layer 2: Convolutional. Output = 10x10x32.
Pooling. Input = 10x10x32. Output = 5x5x32.
Layer 3: Fully Connected. Input = 800. Output = 560.
Layer 4: Fully Connected. Input = 560. Output = 320
Layer 5: Fully Connected. Input = 320. Output = 120.
Layer 5: Fully Connected. Input = 120. Output = 43.

After 79 epochs, validation accuracy is 0.953, and test accuracy = 0.929. The validation accuracy arrives 0.95 for five times, but test accuracy is 0.929, meaning overfitting.
#### Increase hidden number of units in convoluted layers or fully connected layers both don’t add any benefits -- actually they degrade the accuracy. Add a fully-connected layer doesn’t improve the results.

6. Add convolutional layers

Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
Activation
Pooling: Input = 28x28x6. Output = 14x14x6
Extra Layer: Convolutional. Input = 14x14x6. Output = 14*14*12.
Activation
Pooling: Input = 14*14*12. Output = 14x14x12    
Layer 2: Convolutional. 14x14x12  Output = 10x10x16.
Activation
Pooling:Input = 10x10x16. Output = 5x5x16.
Flatten: 5x5x16 --> 400
Layer 3: Fully Connected. Input = 400. Output = 120.
Activation
Layer 4: Fully Connected. Input = 120. Output = 84.
Activation
Layer 5: Fully Connected. Input = 84. Output = 43.

After 82 epochs, validation accuracy reaches 0.942. It enters plateau as early as Epoch=30, and lingers between 0.93 and 0.05.
I also add two more convolutional layers, which are completely no good. The validation accuracy keeps at low levels (<87%)

7. Change egularization using drop out
When dropouts are inserted between fully connected layers, the starting accuracy becomes extremely low (0.059%). I was going to give up the structure, and then found the progressing is really efficient for each epoch. Usually in previous experiments after 10 to 20 epochs, , the validation accuracy sways between 0.9 and 0.94 and changes between epochs are as big as 0.3, and so the validation improvement trend is not obvious. In this test, the range of the “back and forth” reduces to be smaller than 0.1. When the validation rates reach 0.95 for 10 times, I ceased the training and test accuracy is 0.929. Except with one experiment, many experiments with stable high validation accuracy (>0.95 for 10 times) lead to test accuracy at 0.929, which makes me doubt that perhaps 0.929 is most possible rate with the current sampling, i.e. test dataset has different distribution from the training and validation dataset, or the current training dataset is severely unbalanced.

8. Batch normalization
From above realizations, I see the validation accuracy can change in a big range or stabalize at a level with no apparent improvement. It is because the model parameters change due to the distributions of outputs in each layer. These parameter change is like a noise to learning in later layers (https://r2rt.com/implementing-batch-normalization-in-tensorflow.html). I gave the batch normalization for several trials, but the result is very bad, time consuming and accuracy not increasing monotonically at all. After 100 epochs, the validation accuracy =0.697. I guess the convolutional layer has too many units, and maybe my implementation of batch normalization is wrong.

#### Batch normalization


def batch_norm_wrapper(inputs, is_training, decay = 0.999, epsilon = 1e-3):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)      
          
#### Lenet with BN

def LeNet(x, is_training):    
    #  Layer 1: Convolutional. Input = 32x32x3. Output = 32*32*32
    #  Pooling. Input = 32*32*32. Output = 16*16*32   
    #  Layer 2: Convolutional. 16*16*32  Output = 16*16*64.
    #  Pooling. Input = 16*16*64.. Output = 8*8*64.
    #  Layer 3: Input: 8*8*64  Output: 8*8*128
    #  Pooling. Input = 8*8*128  Output = 8*8*128.
    #  Flatten. Input = 8*8*128. Output = 8192.
    #  Fully connected: 8192 -> 2048 -> 512 -> 43    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 32), mean = mu, stddev = sigma))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME')
    
    conv1_bn = tf.contrib.layers.batch_norm(conv1, data_format='NHWC', center=True, scale=True, is_training=is_training)
    
    conv1_bn = tf.nn.relu(conv1_bn)
    conv1_bn = tf.nn.max_pool(conv1_bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1_bn, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2_bn = tf.contrib.layers.batch_norm(conv2, data_format='NHWC', center=True, scale=True, is_training=is_training)

    conv2_bn = tf.nn.relu(conv2_bn)

    conv2_bn = tf.nn.max_pool(conv2_bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 512), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(512))
    conv3   = tf.nn.conv2d(conv2_bn, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    conv3_bn = tf.contrib.layers.batch_norm(conv3, data_format='NHWC', center=True, scale=True, is_training=is_training)
    conv3_bn = tf.nn.relu(conv3_bn)    

    conv3_bn = tf.nn.max_pool(conv3_bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc0   = flatten(conv3_bn)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(8192, 2048), mean = mu, stddev = sigma))
    fc1   = tf.matmul(fc0, fc1_W) 
    print("fc1 before bn", fc1.shape)
    fc1_bn = tf.contrib.layers.batch_norm( fc1, center=True, scale=True, is_training=is_training)
    
    fc1_bn    = tf.nn.relu(fc1_bn)
    fc1_bn = tf.nn.dropout(fc1_bn, keep_prob3)
    print("fc1_bn shape", fc1_bn.shape)

    fc2_W  = tf.Variable(tf.truncated_normal(shape=(2048, 512), mean = mu, stddev = sigma))
    fc2    = tf.matmul(fc1_bn, fc2_W) 
    print("fc2 shape", fc2.shape)
    fc2_bn =  tf.contrib.layers.batch_norm( fc2, center=True, scale=True, is_training=is_training)
    fc2_bn    = tf.nn.relu(fc2_bn)
    fc2_bn = tf.nn.dropout(fc2_bn, keep_prob4)

    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2_bn, fc3_W) + fc3_b
    
    return logits            
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][/new-image/c1.jpg] 
![alt text][/new-image/c11.jpg] 
![alt text][/new-image/c14.jpg] 
![alt text][/new-image/c15.jpg] 
![alt text][/new-image/c18.jpg] 
![alt text][/new-image/c20.jpg]
![alt text][/new-image/c23.jpg] 
![alt text][/new-image/c24.jpg] 
![alt text][/new-image/c27.jpg]

They are cut with ROI and resized to 32*32*3.
![alt text](/examples/newimages.png)

The 5th and 6th image might be difficult to classify because they are tilted and not facing to the front. The last one is also difficult because it has a line of text over it.
The 3rd is also not easy, since there are a lot of noises around the board. The 8th is not easy because after resizing/shrinking, the meaning part is not clear.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way          | Right-of-way  								| 
| Pedestrians  			| Pedestrians   								|
| Stop  	      		| Stop          				 				|
| Speed limit (30km/h)	| Speed limit (30km/h)  						|
| Dangerous curve to the right| Speed limit (120km/h)	            	|
| Road narrows on the right |General caution                            |
| General caution       | General caution                               |
| Slippery road         | Slippery road                                 |
| No vehicles           | Speed limit (30km/h)	                        |

The model was able to correctly guess 6 of the 9 traffic signs, which gives an accuracy of 67%. This compares favorably to the accuracy on the test set of 0.944.
The 6th image, 'Road narrows on the right' is  easy to recognized as 'General caution', since they are very similar.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in step 3 of the Ipython notebook.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The first hidden layer is more apparent in the feature map. They are the outer boundary of the sign, and the inner shapes.
The second hidden convolution layer shows more simple shapes, like two parallel tilted lines in FeatureMap 10, a tilted line in FeatureMap 26, and a vertical line in FeatureMap 12.