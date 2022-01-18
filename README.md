# Meta-Learning: Learning To Learn

![figure 1](https://user-images.githubusercontent.com/80465899/149936332-ae0e26a2-4601-4a4b-8a6f-208218fc0fa1.png)

## After completing this article, you will know:
- The motivation for meta-learning
- What is meta-learning?
- Why is meta-learning important?
- Example scenario
- Common approaches & Applications

# MOTIVATION
The primary objective in meta-learning is learning from very small amounts of data.

## What if you don't have large data sets? For example, you are in a domain such as
- Robotics
- Medical Imaging
- Translation for rare languages
- Recommendation systems

## What if you want a general-purpose AI system in the real world?
For example, the system needs to continuously adapt and learn on the job. Whereas learning each thing from scratch won't cut it.

This is where meta-learning actually comes in handy.

# WHAT IS META-LEARNING?
Meta-learning, or learning to learn, is the science of systematically observing how different machine learning approaches perform on a wide range of learning tasks, and then learning from this experience, or meta-data, to learn new tasks much faster than otherwise possible.

Meta-learning intends to learn the learning process very fast and it learn one task and learn to apply it to other tasks.

"""A meta-learning model will be capable of well adapting or generalizing to new tasks and environments that have never been encountered during training time."""

For example, you are training a binary classifier model in machine learning to classify either a fox or dog from a given image. But if you pass random images of a cat the model predicts it is either a fox or dog whichever has high probabilities. So the model won't be able to learn from the new class images saying it's a cat. But in meta-learning, a classifier trained on non-cat images can tell whether a given image contains a cat or not after seeing some cat pictures.

# Why is meta-learning important now?
Machine learning algorithms have some challenges such as if you are in a domain where you need large datasets for training. Thus, training machine learning models with a very small amount of data is sometimes not effective and with the help of meta-learning, you can achieve the results. Thus applying meta-learning into machine learning improves the following.

- Faster AI systems
- More adaptable to environmental changes
- Generalizes to more tasks
- Optimize model architecture and hyper-parameters

Consider a scenario we have a problem case and associated with the problem, we also have a large dataset. Finally, we are ready to apply several machine learning techniques to obtain some results from the data. While performing we may end up with the following issues.

Training large datasets result in high operational costs due to many experiments during the training phase and take a long time to find the best model which performs the best for a certain dataset.

Meta-learning can help machine learning algorithms to tackle the above challenges by optimizing and finding learning algorithms that perform better.

# Two perspectives of meta-learning
## Mechanistic view
A deep neural network model that can read in an entire dataset and make predictions for new data points. Also training this network uses a meta-data set, which itself consists of many datasets, each for a different task. Finally, this view makes it easier to implement meta-learning algorithms.

##  Probabilistic view
Extract prior information from a set of (meta training) tasks that allow efficient learning of new tasks. Also learning a new task uses this prior and (small) training set to infer the most likely posterior parameters. Finally, this view makes it easier to understand meta-learning algorithms.

# Example Scenario
![figure 2](https://user-images.githubusercontent.com/80465899/149936356-c77ad2e2-0bdb-4a0e-a105-ebe4b71f495d.png)

Let's take a look at the above image on the left side for some seconds. Now try to classify the image on the right side, whether it was drawn by Braque or Cezanne.

If you guessed Braque, then you are right!

## How did you accomplish this?
Even with the very small amount of data you are able to classify the image. So how could you accomplish this?

Through previous experience.

## How might you get a machine to accomplish this task?
You can extract feature information from the data and train a model, there are several ways to extract features and some of them are listed below.
- Modeling image formation
- SIFT features, HOG features + SVM
- Fine-tuning from ImageNet features
- Domain adaptation from other painters

By combining minimal human experience and extracting in-depth features from data-driven experience results in greater success.

To achieve this, and be able to learn the learning process, we will implement meta-learning.

# WORKFLOW
In general, a meta-learning algorithm is trained with outputs (i.e. the model's predictions) and metadata of machine learning algorithms. After training, its skills are tested and used to make final predictions.

Meta-learning covers tasks such as observing the performance of different machine learning models about learning tasks and learning from metadata and finally performing faster learning processes for new tasks.

For [example](https://www.kdnuggets.com/2020/03/few-shot-image-classification-meta-learning.html), we may want to train a model to label different breeds of dogs.

We first need an annotated data set. Then various ML models are built on the training set. They could focus just on certain parts of the dataset. This meta-training process is used to improve the performance of these models. Finally, the meta training model can be used to build a new model from a few examples based on its experience with the previous training process.

![figure 3](https://user-images.githubusercontent.com/80465899/149936382-27ddc890-7afc-49c5-a3e9-2c9d78e3f438.jpeg)

Image caption --> We evaluate the meta-learning model on Labradors, Saint-Bernards, and Pugs, but we just train on every other breed.

NOTE: We are not diving too deep into the workflow. Instead, this article gives you an easy understanding, to get started with meta-learning.

# COMMON APPROACHES
There are three common approaches:

## Model-Based
Model-based meta-learning models update their parameters rapidly with a few training steps, which can be achieved by its internal architecture or controlled by another meta-learner model. Some model-based networks are Memory-Augmented Neural Networks and Meta networks.

## Metric-Based
The core idea in metric-based meta-learning is similar to nearest neighbor algorithms, in which weight is generated by a kernel function. It aims to learn a metric or distance function over objects. The notion of a good metric is problem-dependent. It should represent the relationship between inputs in the task space and facilitate problem-solving. Some metric-based networks are Convolutional Siamese Neural Networks, Matching Networks, Relation Networks, and Prototypical Networks.

## Optimization-Based
What optimization-based meta-learning algorithms intend for is to adjust the optimization algorithm so that the model can efficiently learn from fewer examples such as LSTM Meta-Learner, Temporal Discreteness, and Reptile.

# APPLICATIONS
## Neural Architecture Search
Neural architecture search (NAS) is a technique for automating the design of artificial neural networks (ANN), a widely used model in the field of machine learning. NAS has been used to design networks that can outperform hand-designed architectures.

## Computer vision & Graphics
Meta-learning based on few-shot learning methods trains algorithms that enable powerful deep networks to successfully learn on small datasets. Applications include Object Detection, Landmark Prediction, Object Segmentation, Image Generation, Video Synthesis, Density Estimation.

## Meta Reinforcement Learning and Robotics
Reinforcement learning is typically concerned with learning control policies that enable an agent to obtain high rewards in achieving a sequential action task within an environment. Meta-learning has proved to be effective in enabling Reinforcement learning.

## Language and Speech
Adapting to new languages with Low-Resource Neural Machine Translation. It learns to translate new language pairs without a lot of paired data. Adapting to new persons with personalizing dialogue agents to adopt dialogue to a person with a few examples.

# PROS & CONS OF META-LEARNING
## PROS
Meta-learning allows machine learning systems to benefit from their repetitive application. If a learning system fails to perform efficiently, one would expect the learning mechanism itself to adapt in case the same task is presented again.

## CONS
This method has serious disadvantages, however. First, the resulting rule set is likely to be incomplete. Second, timely and accurate maintenance of the ruleset as new machine learning algorithms become available is problematic. As a result, most research has focused on automatic methods.

# CONCLUSION
There has been an exponential growth in meta-learning algorithms and it's easy to confuse one with its related fields. With the help of taxonomy and broad classification which focuses on functioning and goal, we are able to clearly identify and benchmark the performance of these algorithms and effectively evaluate their applications.

# CREDITS & REFERENCES
[Meta-Learning: from Few-Shot Learning to Rapid Reinforcement Learning](https://sites.google.com/view/icml19metalearning)

[Few-Shot Image Classification](https://www.kdnuggets.com/2020/03/few-shot-image-classification-meta-learning.html) with Meta-Learning

[Meta-Learning: Learning to Learn in Neural Networks](https://medium.com/analytics-vidhya/meta-learning-learning-to-learn-in-neural-networks-843f10408493)
