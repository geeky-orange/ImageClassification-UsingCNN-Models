# ImageClassification-UsingCNN-Models 

<h2> Description </h2>



Though Convolutional Neural Networks (CNN)
have gained a high accuracy in image classification,
the efficiency of the commonly used architectures, including: VGGNet, DenseNet are sacrificed for their
accuracy. In this project, several CNN models will
be built to categorize photos of 17 distinct types of
flowers. These models will then be trained, compared,
and optimized with the ultimate aim of introducing
enhancements to boost model efficiency while keeping
a respectable accuracy rate in the hopes of targeting
lightweight AIoT systems.

<h2> Dataset Description </h2>

#
The “Oxford 17 flowers dataset” contains 17 categories of flowers each with 80 images, in total 1360
images. Given this relatively small dataset, we shall
consider varying model architectures and their ability
to learn through the comparison of performance metrics

<h2> Models Used </h2>

I selected the following as our baseline models

1. DenseNet-121
2.  ResNet-50
3. AlexNet
4. VGG-16

 Each
of the model will be trained and optimized for a
certain set of hyper-parameters first. Then, their
performance will be contrasted in terms of the number
of FLOPS, test accuracy, and validation accuracy.
The models that perform the best will be selected for
more improvements as our ultimate model.

<h2>Methodologies</h2>

We first evaluate several activation functions and hyperparameters to be used in the
training models, after dividing the images in each class
into sub training set, sub validation set, and sub test
set and apply these modifications to each of the models in concern. <br>
<b> In order to optimize the performance of each model.
We would test the efficiency and accuracy of the models with different hyperparameters and activation functions. Our goal is to balance between the accuracy and
efficiency and suggest a model with the modifications.</b>

<h2> Experimentation </h2>

We have applied <b>Stochastic gradient descent (SGD) </b>
for training the models. To ensure a fair comparison on
the efficient, we will first a decide universal batch size
we will apply in all the models, and optimize the model
based on initial learning rate and number of epochs and
reached the optimized accuracy by using the mythology
of grid search.

<h3> Batch size </h3>
<hr>
The batch size is a hyper-parameter that specifies
how many samples must be processed before the internal model parameters are updated. We have opted
for the universal batch size of 10 for all the models except for VGG as it yield almost the best performance
among the models after testing. A batch size of 32 is
used for VGG as it can achieve optimal accuracy. Different batch sizes are used to facilitate fair comparison
among the models to show their optimal results.<br>

<p align='center'>
<img src="./images/Screenshot%202023-01-22%20at%207.17.58%20PM.png" alt="drawing" style="width:250px; display:block; margin-left:auto; margin-right:auto"/>
<br>
</p>

<h3> Initial learning rate </h3>
<hr>

The learning rate is a tuning parameter that controls
the step size at each iteration. For AlexNet(Figure
2), ResNet(Figure 3), DenseNet (Figure 4) and VGG
(Figure 5), we have chose 10−3
, 10−4
, 10−3 and 10−4 as
the learning rate respectively as they yield the highest
test accuracy and average verification accuracy after
training as shown in the figures.

<p float="left" align="center">
<img src="./images/Screenshot 2023-01-22 at 7.31.07 PM.png" alt="graph" style="width:20%; "/>

<img src="./images/Screenshot 2023-01-22 at 7.55.14 PM.png" alt="graph" style="width:20%;"/>

<img src="./images/Screenshot 2023-01-22 at 7.55.34 PM.png" alt="graph" style="width:20%; "/>

<img src="./images/Screenshot 2023-01-22 at 7.55.52 PM.png" alt="graph" style="width:20%;"/>

</p>


<h3> Numbers of epochs </h3>
<hr>


Number of epochs means the number of times the
model are trained against the entire testing set of the
dataset. In Figure 6, we have selected all four models
Since the size of the dataset is relatively small and we
do not requires large number of epochs for classification, as it will easily overfit if we opt for a epochs over
50.
Therefore, we opted for 40, 27, 43 and 25 for VGG,
ResNet, DenseNet and AlexNet respectively.



<h3> Accuracy </h3>
<hr>

All four baseline models are converged. Table 1
has summarized the testing and validation accuracy
and number of parameters of all four baseline models that we have selected.

<p float="left" align="center">
<img src="./images/Screenshot 2023-01-22 at 8.02.53 PM.png" alt="graph" style="width:20%; "/>

<img src="./images/Screenshot 2023-01-22 at 8.03.12 PM.png" alt="graph" style="width:20%;"/>

</p>

DenseNet-121 perform the
best among the four with highest validation accuracy of 83.32% and the lowest number of parameter of 7.98M.
Interestingly, the well-known RetNet and DenseNet do
not preform well in this experiment as the testing and
validation accuracy of these models are relatively lower
than AlexNet and VGG. We will continue to optimise
these two models and aim for getting a better accuracy.
It is worth to note that the testing accuracy of
AlexNet reaches 100.00% on only 35 epochs. We believe that through reducing over-fitting in AlexNet, the
model will reach a higher validation accuracy and it
may become compatible with DenseNet-121.

<p align='center'>
<img src="./images/Screenshot%202023-01-22%20at%208.06.09%20PM.png" alt="drawing" style="width:250px; display:block; margin-left:auto; margin-right:auto"/>
<br>
</p>


<h2> Improvements </h2>

We try to two methods that may
increase the validation accuracy and reduce the number of parameter in our model, which includes layer
dropout and model pruning. We use AlexNet, the
model that we determined that it will have a high potential for improvement, with batch size of 10 and initial learning rate as 103 for optimisation. We will fix
the hyperparameters in order to maintain the network
structure during the optimisation process and have a
fair comparison on the optimisation result.
<BR>
<h3>  Reduce over-fitting: Layer dropout </h3>
<hr>


To reduce over-fitting in AlexNet, we have applied
layer dropout, a regularization method that approximates training a large number of neural networks with
different architectures in parallel by dropping out nodes
randomly with probability p on selected layers in the
network [3] as show in Figure 8.

<p align='center'>
<img src="./images/Screenshot%202023-01-22%20at%2008.11.28%20PM" alt="drawing" style="width:250px; display:block; margin-left:auto; margin-right:auto"/>
<br>
</p>


