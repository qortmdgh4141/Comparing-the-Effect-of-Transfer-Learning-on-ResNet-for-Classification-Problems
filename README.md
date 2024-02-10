# 👨‍👨‍👦‍👦🏻  Comparing the Effect of Transfer Learning on ResNet for Classification Problems  
<br/>
  
### 1. &nbsp; Experiment Overview <br/><br/>

- _This experiment aims to train and validate an Artificial Intelligence model for age classification of specific individuals, utilizing a Korean facial dataset. The dataset employed is a publicly available Korean facial image dataset from AI-Hub._ <br/>

- _Additionally, the experiment is based on one of the key models in deep learning, ResNet50. ResNet, or Residual Network, addresses the issue of gradient vanishing in deep neural networks through residual connections. This architecture facilitates efficient training of deep networks and has shown exceptional performance in image classification and recognition tasks._ <br/>

- _The primary focus of this experiment is to compare the performance of various state-of-the-art deep learning techniques. To this end, improvements in the ResNet architecture proposed in various papers, methods of data preprocessing, and learning strategies have been considered. While maintaining the basic structure of the ResNet50 model, various optimization techniques, including transfer learning, fine-tuning, and learning rate scheduling, will be applied in this experiment._ <br/>

- _The outcomes of this experiment are expected to contribute significantly to the advancement of facial recognition technology, with potential applications in security, marketing, and personalized service delivery._ <br/><br/>

### 2. &nbsp; Dataset Introduction <br/><br/>

- _The dataset used in this experiment is a Korean facial image dataset provided by AI HUB. This dataset comprises data collected from over 1,000 direct family members across generations, covering a wide age range from 0 to 80 years._ <br/>

- _One common approach for age prediction involves using regression models. However, in this experiment, we opted for a classification model. This strategic decision was made to enhance the model's learning efficiency and prediction accuracy. Specifically, categorizing by age groups allows the network to more clearly learn the characteristics of specific age brackets. We believe this approach will improve prediction accuracy compared to regression models._ <br/>

- _For this purpose, additional preprocessing was performed during the dataset's labeling process. The ages were classified into eight categories based on age groups: 0-9 years, 13-19 years, 20-29 years, 30-39 years, 40-49 years, 50-59 years, 60-69 years, and 70 years and above._ <br/>

<p align="center">
  <img width="50%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig1.png?raw=true">
  <br>
  <em> Figure 1) Classified Korean Facial Data into 8 Categories</em>
</p> 

- _The dataset used in our study has certain limitations._ <br/>
  1) _Firstly, the amount of data is quite limited for conducting sufficient training._ <br/>
  2) _Secondly, there is an imbalance in the distribution among classes. Such a limited quantity of data or a distribution biased towards specific classes can lead to overfitting problems during the learning process._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig2.png?raw=true">
  <br>
  <em> Figure 2) Age-wise Distribution of Korean Facial Data Labels
</p> 

- _To overcome these issues, I have designed experiments by considering various training strategies that have been validated in previous research._ <br/> <br/>
    
### 3. &nbsp; Experimental Methodology <br/><br/>

- #### _Transfer Learning_
  _The primary hypothesis of this experiment is that models applying transfer learning, based on a small number of epochs, will demonstrate superior performance under training conditions with limited datasets. This hypothesis is grounded in the transfer learning and fine-tuning techniques presented in "Deep Residual Learning for Image Recognition" (CVPR, 2016) and "Best Practices for Fine-tuning Visual Classifiers to New Domains" (ECCV, 2016)._ <br/>

- #### _Efficient Training_
  _In this experiment, we adopted the Efficient Training methodology introduced in "Bag of Tricks for Image Classification with Convolutional Neural Networks" (CVPR 2019), implementing a strategy of initializing the γ (gamma) parameter in Batch Normalization layers to zero. The paper noted that applying this method to the ResNet architecture, which includes residual connections, improved network stability in the early stages of training. In our experiment, this initialization technique played a significant role in enhancing the model's convergence rate and overall stability of the learning process._ <br/>

- #### _Training Refinement_
  _We applied the Cosine Learning Rate Decay technique, mentioned in "Bag of Tricks for Image Classification with Convolutional Neural Networks" (CVPR 2019). This technique schedules the learning rate to decrease slowly initially, almost linearly in the middle, and then slowly again towards the end. Compared to Step Decay, this approach induces faster convergence early in training and allows for finer weight adjustments later, thereby refining the entire learning process. Our experiment, comparing Step Decay and Cosine Learning Rate Decay, showed that the latter had a positive impact on the stability of the overall learning process._ <br/>

- #### _Model Tweaks_
  _To enhance the performance of the ResNet architecture, "Bag of Tricks for Image Classification with Convolutional Neural Networks" (CVPR 2019) proposed modifications such as ResNet-C with a changed Input Stem structure, and ResNet-B and ResNet-D with altered structures in Stage 4 Downsampling Block. In our experiment, we trained a combined ResNet-BCD model, which integrates ResNet-B, ResNet-C, and ResNet-D structures, and then compared its performance with the original ResNet model._ <br/>

- #### Ultimately, in this experiment, we trained and analyzed four different models applying variations of the ResNet architecture and transfer learning strategies to the given dataset:
   - _ORG-Model: A model based on the original ResNet-50 architecture._ <br/> 
   - _TL-Model: A model that applies transfer learning to the original ResNet-50 architecture._ <br/> 
   - _BCD-Model: A model incorporating modifications of the ResNet-B, ResNet-C, and ResNet-D architectures._ <br/> 
   - _TL-BCD-Model: A model that applies transfer learning to the combined ResNet-B, ResNet-C, and ResNet-D modified architectures._ <br/> 

<p align="center">
  <img width="50%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/Table1.png?raw=true">
  <br>
  <em> Table 1) Comparison of Training Approaches Across Different ResNet Models
</p> 

### 4. &nbsp; Experimental Settings <br/><br/>

- _Due to hardware constraints, specifically limited GPU memory, our experiment could not accommodate the batch size of 256 and learning rate of 0.1 as used in the referenced papers._ <br/>

- _We therefore adjusted our approach to a batch size of 32 and correspondingly modified the learning rate to 0.01._ <br/>

- _However, for models initiated with training from scratch, we maintained the learning rate at 0.1 to prevent underfitting._ <br/>

<p align="center">
  <img width="50%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/Table2.png?raw=true">
  <br>
  <em> Table 2) Summary of Experimental Settings
</p> 
    
### 5. &nbsp; Research Results  <br/><br/>
    
- _Previous studies have suggested that reducing the number of epochs when training with small datasets may serve as a solution to prevent overfitting. Our experimental outcomes reflect this, as none of the four models displayed signs of overfitting._ <br/> 

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig3.png?raw=true">
  <br>
  <em> Figure 3) Loss Curves for Training and Validation Datasets 
</p> 

- _However, our analysis revealed that in complex network structures, a reduced number of epochs led to another challenge: underfitting._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig4.png?raw=true">
  <br>
  <em> Figure 4) Accuracy Curves for Training and Validation Datasets (1)
</p> 

- _As a solution, integrating additional training strategies, such as fine-tuning, alongside limiting the number of epochs, has proven to be highly effective, as demonstrated by the enhanced performance of the TL-Model and TL-BCD-Model._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig5.png?raw=true">
  <br>
  <em> Figure 5) Accuracy Curves for Training and Validation Datasets (2)
</p> 

### 6. &nbsp; Reflections and Future Research Directions Post-Experiment <br/><br/>

- _After applying the image preprocessing techniques recommended in the "Bag of Tricks for Image Classification with Convolutional Neural Networks" paper, we observed underfitting in all four models. This led us to omit these methods in the final stages of our experiment, and it has informed my perspective that future research should meticulously adjust preprocessing techniques to suit the specific type of dataset used._ <br/>

- _Unexpectedly, there was no notable performance disparity between the ResNet Original and its BCD variant architectures. We discovered that factors such as the learning rate scheduler, epoch count, dataset diversity, and initialization methods exerted a greater influence on model performance than the complexity of the model structure itself. This experience has reinforced the notion that careful attention to hyperparameter settings is just as crucial as model design._ <br/>

<p align="center">
  <img width="50%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig6_.png?raw=true">
  <br>
  <em> Figure 6) Accuracy Curves for Test Dataset
</p> 

- _In this experiment, we did not specifically address the issue of data imbalance. Should further experiments be conducted, we would consider the impact of data imbalance on model training and contemplate the use of a Focal-Loss function instead of Cross-Entropy to potentially build a more optimized model._ <br/>

<p align="center">
  <img width="50%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig7.png?raw=true">
  <br>
  <em> Figure 7) Comparison of Loss Functions: Cross-Entropy vs Focal Loss 
</p> 
    
### 7. &nbsp; Additional Experiments <br/><br/>

- _In our study, the TL-BCD model, which showed superior performance, validated its effectiveness during the internal validation phase. However, external validation is critical for generalization checks before deploying a model in actual applications. Therefore, we collected images of Korean celebrities not present in the AI-Hub's Korean facial dataset from the internet and carried out external validation. The results, as depicted in the figure below, confirmed that the model also exhibits excellent performance in external validation scenarios._ <br/>

<p align="center">
  <img width="90%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/fig8.png?raw=true">
  <br>
  <em> Figure 8) External Validation Using Images of Korean Celebrities Across Various Age Groups
</p> 
    
<br/>
 
--------------------------
### 💻 S/W Development Environment
<p>
  <img src="https://img.shields.io/badge/Windows 10-0078D6?style=flat-square&logo=Windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google Colab-black?style=flat-square&logo=Google Colab&logoColor=yellow"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-FF9900?style=flat-square&logo=PyTorch&logoColor=EE4C2C"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=blue"/>
</p>

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; AI Hub Dataset : Korean Face Image <br/>
