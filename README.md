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
network [3] as show in Figure below.

<p align='center'>
<img src="./images/Screenshot 2023-01-22 at 8.11.28 PM.png" alt="drawing" style="width:250px; display:block; margin-left:auto; margin-right:auto"/>
<br>
</p>

In order to tune the optimal probability for model
estimation, we minimize a loss function. For this linear layer, we will look at the ordinary least square loss Bernoulli(p). This means square loss the value is equal
to 1 with probability p and 0 otherwise. Since in our
case the drop out rate. After finding the relationship
between gradient of drop out network and regular network we find out that drop out network is equivalent to
the regularised Network which means that minimising
the Dropout loss is equal to minimising the regularised
network. Therefore, since the regularization parameter
is p(1-p) in equation therefor maximum is at p = 0.5

<p align='center'>
<img src="./images/Screenshot 2023-01-22 at 8.20.17 PM.png
" alt="drawing" style="display:block; margin-left:auto; margin-right:auto"/>
<br>
</p>

With layer dropout, the test accuracy of model with
layer dropout have successfully reduced from 100.00%
to 98.42%, as shown in Figure 9. 

<p align='center'>
<img src="./images/Screenshot 2023-01-22 at 8.22.53 PM.png
" alt="drawing" style="width: 250px; display:block; margin-left:auto; margin-right:auto"/>
<br>
</p>

In addition, the validation accuracy has successfully increased from 86.03% to 90.44%, which is a 4.11% increase. This indicates the layer dropout have successfully implemented to reduce over-fitting of the model and we will include layer dropout in our finalised model.

<h3>  Reduce parameters: Pruning </h3>
<hr>

Models can be pruned effectively to minimize their
size. Pruning is a very straightforward technique,
which involves removing unnecessary branches from
models, especially those rarely utilized or insignificant
ones, in order to save valuable storage space, reduce
activation sizes, and reduce computational complexity,
thus reducing inference time.
Due to the fact that most neurons and connections
will remain in the model, it should have minimal impact on accuracy. We decided to test out this method
on AlexNet with random pruning with a probability of
10%, on two of the Conv2D layers, which are the first
two layers of AlexNet. The result was unexpected.



<p align='center'>
<img src="./images/Screenshot 2023-01-22 at 8.26.00 PM.png
" alt="drawing" style="width: 250px; display:block; margin-left:auto; margin-right:auto"/>
<br>
</p>

In spite of the fact that the test accuracy has not
changed significantly after pruning, 98.42% before and
98.6% after pruning, the validation accuracy has decreased drastically. Prior to pruning, the model was
90.44% accurate, but after pruning, it drops significantly to 66.18%, a decrease of 24.34%. We would not
use this method to optimize our AlexNet model because the significant drop in validation accuracy is not
worth the tradeoff.


<h2> Conclusion </h2>

As part of the evaluation of DenseNet-121, ResNet50, AlexNet, and VGG-16 models, batch sizes, learning
rates, and epochs are adjusted. Using the current optimization methods available, we investigated AlexNet
and DenseNet further to determine if any model could
be utilized for an AIoT application. Our two optimization methods included layer dropout, which reduces
overfitting and has minimal impact on accuracy. As
pruning would significantly reduce the accuracy of validation set, we did not decide to opt for it.

