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

<p float="left">
<img src="./images/Screenshot 2023-01-22 at 7.31.07 PM.png" alt="graph" style="width:25%; "/>

<img src="./images/Screenshot 2023-01-22 at 7.31.07 PM.png" alt="graph" style="width:25%;"/>

<img src="./images/Screenshot 2023-01-22 at 7.31.07 PM.png" alt="graph" style="width:25%; "/>

<img src="./images/Screenshot 2023-01-22 at 7.31.07 PM.png" alt="graph" style="width:25%;"/>

</p>

<!-- <p align="center" float="left">

</p> -->


<!-- <p display:'inline'>

</p> -->



