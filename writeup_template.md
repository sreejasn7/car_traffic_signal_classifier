# **Traffic Sign Recognition** 

## SUMMARY


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/train_images.png "Training Set"
[image2]: ./pics/plot_histogram_train.png "Training Set Histogram"
[image3]: ./pics/histogram_equalized.png "Histogram Equalisation"



### Writeup / README
My code is in Traffic_Sign_Classifier.ipynb


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy and basic python library to calculate summary statistics of the traffic
signs data set:

    * The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.

* Shows some rain images with labels [here][image1]
* Analyzed the histogram of the images as there some of the images are not clear to predict . [histogram of train set][image2]

### Design and Test a Model Architecture

#### PREPROCESING

- Working on RGB images . 3 layer images. COnvert to YUV before Histogram Eq.
- Histogram Equalization done to enhance contrast of low pixel intensity images. This is done in equilizer function. Equalization is done on Y and U and V is kept as such. (This idea was taken from the PDF shared along with the assignment)

- Generated More data:
   - More data was generated to increase the accuracy for test set . (was at o% accuracy. SO had to take up and add more data)
   - Generated More data under the section # SECTION _ MORE DATA (In[8] in ipython notebook)
   - Min of 1000 samples are set for every class in the training set . 
   - The images that are to be added are selected randomly from the training set and augumented by only slight variations in rotation . This was necessary as the test images or web images all had slight tilt. The agumentaiton was done by rotate_image function.

- Normalisation .
  - Coded under the section Section Normalization (In - 12 in ipython notebook)
  - The normalisation for achieved for train , validate  , test set.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|		Description								| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image								| 
| Convolution 5x5   	| 1x1 stride, same padding, outputs 28*28*6		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5  	 	| Stride 1 , valid padding , output 10x10x6		|
| RELU					|		      									|
| Max pooling 			|2x2 stride , output 5x5x16 					|
|FLatten				|Input 5x5x16, output 400						|
|Fully connected		|Input 400, output 120							|
|Fully connected		|Input 400, output 120							|
|RELU					|												|
|Fully Connected 		|Input 120 ,output 84							|
|RELU					|												|
|Fully connected 		|Input 84 output 43								|


#### 3. Training Model Parameters

To train I used. 
EPOCHS = 40
BATCH_SIZE = 128
Learnign rate = 0.001. 

I think increasing the epochs sometimes causes overfitting.I am not sure here.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.9%
* test set accuracy of 92.1 %

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? 
 - I choose the default architecute of LeNet 5 
* What were some problems with the initial architecture?
- Main problem with the sample code was I could not use 3 layer images.Changed the filter size. And Lenet supports multichannel 32 x 32.

* How was the architecture adjusted and why was it adjusted?
- Used only max pool and relu activation functions.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? 
THe only chosed hyperparameter by me was Epoch = 40. This was necessary to increase the accuracy on the train set. 

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Different German test images that are found on web are included in the test_images folder.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road    		| Priority road									| 
| General caution		| General caution								|
| No entry 				| No entry										|
| Roundabout mandatory	| Roundabout mandatory			 				|
| Double Curve			| Double Curve      							|
| Speed limit (30km/h)	| Speed limit (30km/h) 							|
| Speed limit (20km/h)	| Right-of-way      							|
| Keep right			| Keep right      								|
| Keep left				| Keep left      								|
| Stop					|Speed limit (20km/h)  							|


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .09					| Priority Road									| 
|  1					| General Caution								|
|  1					| No entry										|
|  1					| Roundabout mandatory				 			|
| .09					| Double Curve      							|





