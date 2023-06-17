# Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems


1. Research Overview

In this study, I train a ResNet50 model using the AI-Hub Korean Face Image dataset to estimate the age of individuals.  In the process, I compare the classification performance of the Original ResNet50 model and the TL-ResNet50 model that applies transfer learning (TL) technique. The main hypothesis of this research is as follows:

"Under the condition of training with a small dataset and a small number of epochs, the model using transfer learning will perform better."

This hypothesis is based on the following, referencing the ResNet-50 model structure from the paper “Deep Residual Learning for Image Recognition" (CVPR, 2016) and the transfer learning and fine-tuning experiments from the paper "Best practices for fine-tuning visual classifiers to new domains" (ECCV, 2016).

I. Model complexity and number of epochs: Overfitting may occur when training complex models with many layers and parameters (e.g., ResNet-50, VGG-19), which can be avoided by training with a small number of epochs. However, training with a small number of epochs may cause another problem: underfitting due to insufficient training on the entire dataset. 

II. Data quantity: Training a model with a small dataset can lead to overfitting due to lack of diversity in the data. This can be addressed through transfer learning and fine-tuning, which leverages the generalization ability of pre-trained model trained with large datasets, and generally performs well even with a small amount of data (diversity) because it performs new tasks (model training) based on the weights of pre-trained model that have trained patterns from a variety of images.

Based on this background, I will compare the performance of estimating (classifying) an individual's age by training the existing ResNet50 model and the TL-ResNet50 model, respectively. I expect that the results will enable the estimation of an individual's age, which can be applied to various fields such as predicting the age of criminals.
