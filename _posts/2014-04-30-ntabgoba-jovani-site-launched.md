---
layout: post
title: "Introduction to Deep Learning"
date: 2014-04-30
---
Deep Learning Overview
Ntabgoba Jovani 
M.Sc. Student

Deep Learning Course    taught by:
Professor Hideki Kozima                                                

14th August 2016


Abstract—This paper is a brief introduction to the theoretical and application of deep learning.  
Keywords—deep learning; machine learning; neural networks

Abstract—This paper is a brief introduction to the theoretical and application of deep learning.  
Keywords—deep learning; machine learning; neural networks
I.	 Introduction 
Deep learning is a machine learning algorithm that uses the concept of neural networks coupled with back propagation to create a series of functions that discover patterns. It mimics the human being’s brain. Just as the brain learns any task by developing connections of neurons so does the computer using deep neural networks.  
Deep learning (DL) does not require human selected independent variables (features) of a pre-training dataset. Through large training datasets, DL is able to auto learn and discover relevant features continously. This continued learning makes it possible for the DL to predict/infer on datasets that it has seen for the first time.
Deep learning is part of continued innovations in the machine learning field that harnesses the computer’s computational power to develop functions that can infer or predict outcomes, given a set of inputs.
II.	Machine Learning
Machine Learning refers to processes of using statistical methods to infer or predict one or more dependent variables based on independent variables. It involves the use of computers to do large computations to discover patterns.
If we define a data set as n rows by p columns. For each row xi = x11 + x12+ …+ x1p, there is a corresponding yi (row name).  Y is a dependent variable and X is an independent variable. 
X = 
Often 2 or 3 datasets of the same object under scrutiny are required, namely training, testing and validation datasets. The training dataset is used to develop a model, a function that approximates the pattern(s) within the dataset. The Testing dataset is used to estimate the error rate, while validation dataset validates consistency of the model when applied to a dataset seen for the first time.




For example, in a classification problem to distinguish whether a given image is a cow or a goat. Values of an image’s pixels will be the vector p of x independent variable and the factor variables (goat or cow) will be the Y dependent variable.
      

From fig 1, the pixel representations of images of a cow and a goat provide a large array that represents all the minute parts of the images.
There are several machine learning approaches to discover patterns from a dataset. Examples are linear regression, decision trees, support vector machines and more. The ultimate goal of these algorithms is to develop a function f: X  Y.  The difference in these algorithms arise from the numerous regression or classification techniques that they each apply.
Selection of which algorithm that best suits a given task, depends on several considerations like;
- When dealing with structured quantitative observations like sales versus marketing budgets. Simpler algorithms that use regression analysis that is easy to interpret and usually works well are preferred.
- When n > p, especially when dealing with gene expression problems. Algorithms like Ridge regression or Lasso that shrink the number of features are preferable. 
- When dealing with classification problems that mainly care about the accuracy of the output and not the intricate behavior of the input-output function, then deep learning is among the preferred algorithms [1].
Machine learning can be classified into two categories, that is; Supervised learning and Unsupervised learning.
A.	Supervised Learning
Supervised Learning is a type of machine learning where the training dataset used to develop a model, already has human labeled Y dependent variables. From above example, each image (x variables) will have initially been classified by humans as either a cow or a goat (y variable).


A.	Un Supervised Learning
Supervised Learning is a type of machine learning where the training dataset’s dependent variables are not prior labeled by a human being. From above example, all image (x variables) will not be labeled in the training dataset.
II.	Deep Learning Theory
Deep learning architecture is a multilayer stack of simple modules, all (or most) of which are subject to learning, and many of which compute non-linear input-output mappings. [3]. 
Deep learning has three major kind of neural networks;
A.	Deep Neural Networks
A standard neural network consists of a series of connected layers of processors called neurons, each using a nonlinear function to produce a sequence of real-valued activation. [4]


A neural network consists of an input layer (color green), that has several connections to the hidden layers (color red) that also lead to an output layer (color blue).
The above figure essentially involves two processes, that is;
Feedforward process
Each connection is given a random small number called a weight wi and an input xi , resulting into zi = wi xi  . Therefore, zi is an input at every neuron of a hidden layer. The summation of all zi inputs to each neuron is activated using a nonlinear function like a Rectified Linear Unit (ReLU) or a Sigmoid function. Each hidden layer neuron produces an output;
yi = f(zi).   These outputs yi are fed into subsequent layers of the hidden layers up to the output layer youtput 
The mathematical transformations due to this distributed addition of weights and activation functions is derived using chain rule of derivatives.
Empirically, the performance of neural networks will differ from task to task. Therefore, to fine tune a neural network you can;
-Increase the number of hidden layers to hundreds or hundreds of thousands layers (making it deeper).
-Increase the number of neurons on each layer (making it wider).




- Change the activation function from ReLU to sigmoid or tangent function or other type of nonlinear function. [7]
Backpropagation
	From feedforward neural network, there is no way of real time comparison of the predicted output with the accurate human labeled training dataset variables.
Backpropagation is a process of recursively comparing the predicted out with the accurate variable, and feeding back their difference into the neural network in a backward direction similar to the feedforward.
	
Where: predicted variable = youtput
                   Accurate variable = human labeled variable in training dataset.
The error,  is propagated backwards throughout the hidden layers, hence gradually adjusting the randomly assigned weights of the connections [3]. 
It is this repeated process of feed forward, comparing the output with the accurate variable and back propagating the error, that finally leads to high performing neural networks. 
As the network gets deeper, wider and as the number of iterations of feedforward and backward propagation gets larger, Neural networks require a lot of computational power. This is the reason why most libraries that deal with deep neural networks also run on Graphical Processing Units (GPUs). 
GPUs have parallel architecture which speeds up vector and matrix computations, reducing the run time from months (if on CPU) to weeks or days (if on GPU).
A.	Convolutional Neural Networks
Convolutional Neural Networks (ConvNets) is a form of neural networks with an architecture of visual cortex ventral path-way inspired by the classic notions of simple cells and complex cells in visual neuroscience [3].



A convolutional layer is implemented by convolving the input with a set of filters, followed by elementwise non-linear function generating feature maps, each unit of a feature map is connected to local patches of the previous layer through a set of weights [8].
ConvNets capitalize on the properties of natural signals; local connections, shared weights, pooling and the use of many layers. A convolution layer detects local conjunctions of features from a previous layer and pooling merges semantically similar features into one [3]. Units of a given feature map in two consecutive convolution layers are connected together with the same weights called a filter bank. 
This notion of equal weights allows pooling to easily merge similar motifs without caring about its precise location. For example, from fig 3, the ConvNet will add all green patches into one motif irrespective whether the test dataset images are of a goat standing, laying down or running. 
On the other hand, Feature maps form local groups of motifs that are often spatially correlated and thus easy to detect. For example, feature maps of the goat’s eye will often appear near to the feature maps of the goat’s nose, irrespective of whether the test dataset images are of a goat facing up or down or sideways.
ConvNets is famous in image recognition following the 2012 ImageNet Large Scale Visual Recognition Challenge, where a ConvNets architecture called AlexNet won the contest [9].
A.	Recurrent Neural Networks
Recurrent Neural Networks (RNN), each neuron is self-connected with a constant weight w, to have a sort of memory. RNNs are well suited for sequential tasks like speech recognition and in natural language processing like predicting the next word in a sentence.


Compared to the Deep Neural Network, an RNN has an extra input w, on each neuron, which is a record of the past activation. Since the input to the RNN is usually in time steps, the memorized weight w, helps to keep the contextualization of the pattern recognition.
I.	Application of Deep Learning 
A.Image Classification
Computer vision is one domain where deep learning registered great success compared to other machine learning techniques. Deep convolutional neural networks have been adopted by Tech industry giants like Google, Facebook and Baidu for image understanding and search. [10].
For example, below are results of applying of IBM’s deep learning computer vision API called Alchemy on fig 1 images. This API combines deep learning with other machine learning algorithms.


From Table 2. The API predicted correctly the image of a cow as a cow at 97% score. However, the API predicted the image of a goat as a sheep at 50% score. 
B.Voice Recognition
Deep Neural networks have also shown great performance in voice recognition.



Table 3 is an example of Google Cloud Speech API [5] that converted my speech to text at 100% accuracy rate, despite my pronunciation being of a non-native English speaker. The API also offers 88% and 86.9% confidence as its certainty of what it heard.
C.Text Analysis
Recurrent neural networks are also have registered impressive performance in text analysis and sentiment analysis. 
Below is a text analysis of part I and II of this paper using Alchemy Language API [6].


Table 4, You notice that the highly relevant words are “data validation consistency” which surprisingly represent the machine learning processes of getting data, trying models that can be valid and then testing for consistency of the model.
The API also offers the Sentiment analysis which turns out to be only positive and neutral.
I.	Deep Learning Libraries
There are several deep learning libraries but the major ones are developed in C++, Python, MATLAB and R programming languages. 
There are several other deep learning algorithms. Some are too
Specific for a given task and others are a bit general purpose.


Acknowledgment 
Acknowledgments and great thanks go to Prof Hideki Kozima for making deep learning course, very interesting.
Great thanks also go to Prof Togashi Atsushi for his invaluable time on discussions, book recommendations and purchases.

References
Gareth James,Daniela Witten,Trevor Hastie &Robert Tibshirani “An Introduction to Statistical Learning with Applications in R” ,Springer, pp 11,17,26
Trevor Hastie,Robert Tibshirani&Jerome Friedman, “Elements of Statistical Learning”,2nd Edition, Springer, 2008, pp 392
Yann LeCun, Yoshua Bengio & Geoffrey Hinton, “Deep learning,” Nature, Vol 521, 2015, pp 438 and pp 439.
Ju ̈rgen Schmidhuber, “Deep Learning in Neural Networks: An Overview”, Technical Report IDSIA-03-14 / arXiv:1404.7828 v4, 2014,pp 4. 
Google API of speech recognition https://cloud.google.com/speech/ 
IBM Watson’s Text Analysi API https://alchemy-language-demo.mybluemix.net/ 
Arno Candel, Jessica Lanford, Erin LeDell, Viraj Parmar & Anisha Arora, “Deep Learning with H2O”, 3rd Edition, 2015, pp 8 
Fan Hu, Gui-Song Xia, Jingwen Hu & Liangpei Zhang “Transferring Deep Convolutional Neural Networks for the Scene Classification of High-Resolution Remote Sensing Imagery”, MDPI AG, Basel, Switzerland, 2015 
Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.; Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.; et al. “Imagenet large scale visual recognition challenge”. Int. J. Comput. Vis. 2015
Yangqing Jia , Evan Shelhamer , Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama & Trevor Darrell “Caffe: Convolutional Architecture for Fast Feature Embedding” OPEN SOURCE SOFTWARE COMPETITION UC Berkeley, pp 1, 2014.





