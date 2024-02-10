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
  _To enhance the performance of the ResNet architecture, "Bag of Tricks for Image Classification with Convolutional Neural Networks" (CVPR 2019) proposed modifications such as ResNet-C with a changed Input Stem structure, and ResNet-B and ResNet-D with altered structures in Stage 4 Downsampling Block. In our experiment, we trained a combined ResNet-BCD model, which integrates ResNet-B, ResNet-C, and ResNet-D structures, and then compared its performance with the original ResNet model. _ <br/>

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

- While the referenced papers used a batch size of 256 and a learning rate of 0.1, our experiment, due to limitations in hardware (GPU) memory resources, adopted a batch size of 32. Consequently, we linearly adjusted the learning rate to 0.01. However, for models not utilizing transfer learning, a lower learning rate led to underfitting, so we set the learning rate at 0.1 for these models. Additionally, due to the limited amount of data in our dataset, overfitting became more frequent as the number of epochs increased. Therefore, we set the number of epochs to a relatively small but appropriate value.

<p align="center">
  <img width="50%" src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/Table2.png?raw=true">
  <br>
  <em> Table 2) Comparison of Training Approaches Across Different ResNet Models
</p> 
    
### 5. &nbsp; Research Results  <br/><br/>
    
- _The objective of this study was to compare the classification performance between the Original-ResNet50 model and the TL-ResNet-50 model with transfer learning (TL) technique applied, using the AI-Hub Korean face image dataset for estimating person's age._ <br/> <br/> 
  
  ```
  def plot_loss_and_accuracy(org_train_loss, trans_train_loss, org_val_acc, trans_val_acc):
      epochs = range(1, len(org_train_loss) + 1)

      plt.figure(figsize=(12, 6))

      # Loss 그래프
      plt.subplot(2, 2, 1)
      plt.plot(epochs, org_train_loss, 'b', label='Orginal-ResNet')
      plt.plot(epochs, trans_train_loss, 'r', label='TL-ResNet')
      plt.title(f'Training Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()

      # Accuracy 그래프
      plt.subplot(2, 2, 2)
      plt.plot(epochs, org_val_acc, 'b', label='Orginal-ResNet')
      plt.plot(epochs, trans_val_acc, 'r', label='TL-ResNet')
      plt.title(f'Accuracy on Validation Data')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()

      plt.tight_layout()

      plt.show()

  # 그래프 출력
  plot_loss_and_accuracy(org_train_loss, trans_train_loss, org_val_acc, trans_val_acc)
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/line_graph.png?raw=true">

- _Analyzing the "Training Loss" graph, we observe that the Original-ResNet50 model experiences underfitting from the 2nd epoch onwards, with the loss value not decreasing gradually. In contrast, the TF-ResNet50 model consistently exhibits a gradual decrease in the loss value even after the 2nd epoch. Moreover, starting from the 1st epoch, the TF-ResNet50 model, which utilizes the pre-trained model's weights as initial values, outputs significantly lower loss values compared to the Original-ResNet50 model._ <br/>
 
- _The superiority of the TF-ResNet model is also evident in the "Accuracy on Validation Data" graph, where the TF-ResNet50 model consistently achieves considerably higher accuracy values than the Original-ResNet50 model across all epochs. Notably, there is a substantial difference in accuracy values between the two models, particularly when reaching the final 10th epoch._ <br/><br/> 

  ```
  def gradientbars(bars, cmap_list):
      # cmap 가중치 설정
      grad = np.atleast_2d(np.linspace(0,1,256)).T
      # 플롯 영역 재설정
      ax = bars[0].axes
      lim = ax.get_xlim()+ax.get_ylim()
      ax.axis(lim)
      # 각 막대에 색 입히기
      max = 0
      for i, bar in enumerate(bars):
          bar.set_facecolor("none")
          x,y = bar.get_xy()
          w, h = bar.get_width(), bar.get_height()
          ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", cmap=cmap_list[i])

          plt.text(x+w/2.0+0.015, h+0.7, "{}%".format(h), fontsize=14, ha='center', va='bottom')

  _, org_test_acc = test(net=org_net, criterion=org_criterion)
  _, trans_test_acc = test(net=trans_net, criterion=trans_criterion)

  df = pd.DataFrame({'Model':['Orginal-ResNet', 'TL-ResNet'], 'Accuracy': [round(org_test_acc*100) , round(trans_test_acc*100)]})

  fig, ax = plt.subplots(figsize=(6,5))
  cmap_color = ['viridis_r', 'YlOrRd']
  gradientbars(ax.bar(df.Model, df.Accuracy), cmap_color)

  plt.title("     < Performance Comparison of Models >     \n", fontsize=18)
  plt.ylabel('Accuracy', fontsize=16)
  plt.ylim([0, 80])
  plt.xticks(fontsize=16)

  plt.show()

  ```
  <p align="center">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/bar_graph2.png?raw=true" alt="bar_graph2" width="480" >&nbsp;&nbsp;&nbsp;&nbsp;
  
- _Finally, I compared the accuracy values of the two models on real test data and found that the TF-ResNet model has about 2x higher accuracy than the Original-ResNet50 model. These results prove that models using transfer learning perform better when training with small datasets and a limited number of epochs._ <br/> <br/> 
  
  ```
  class CustomDataset(Dataset):
      def __init__(self, image_paths, transform = None):
          self.image_paths = image_paths
          self.transform = transform

      def __len__(self):
          return len(self.image_paths)

      def __getitem__(self, idx):
          image_path = self.image_paths[idx]
          img = Image.open(image_path)

          channels = len(img.getbands())
          if channels == 4:
              img = img.convert("RGB")  # 알파 채널을 제외하고 RGB로 변환

          if self.transform:
              img = self.transform(img)
          return img

  def real_img_test(net, loader):
      net.eval()

      pred_list = []
      for i, batch in enumerate(loader):
          imgs = batch
          imgs = imgs.cuda()

          with torch.no_grad():
              outputs = net(imgs)
              _, preds = torch.max(outputs, 1)

          pred_list.append(preds.item())

      return pred_list

  label_to_age = {
      0: "Kids \n (0~9 years old)",
      1: "Young Adults \n (Teens : 13~19 years old)",
      2: "Young Adults \n (Twenties : 20~29 years old)",
      3: "Young Adults \n (Thirties : 30~39 years old)",
      4: "Middle-aged Adults \n (Forties : 40-49 years old)",
      5: "Middle-aged Adults \n (Fifties : 50-59 years old)",
      6: "Old Adults \n (Sixties : 60-69 years old)",
      7: "Old Adults \n (Seventies and Older : 70~)"
  }

  real_test_transform = transforms.Compose([
      transforms.Resize(128),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  ImageFile.LOAD_TRUNCATED_IMAGES = True

  image_paths = ["/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img1.png", 
                 "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img2.png", 
                 "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img3.png",
                 "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img4.png",
                 "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img5.png", 
                 "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img6.png",
                 "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img7.png",
                 "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/img8.png"]

  names = ["[ Chu Sarang ]", "[ Jang Wonyoung ]", "[ Me ]\n(Baek Seungho)", "[ Swings ]", "[ Lee Jungjae ]", "[ Ryu Seungryong ]", "[ Na Moonhee ]", "[ Kim Youngok ]"]

  real_dataset = CustomDataset(image_paths, real_test_transform)
  real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=1, shuffle=False)

  model_path = "/content/drive/MyDrive/Colab Notebooks/Github_Repository/TL-ResNet/TL-ResNet_model.pt" # 모델 파일 경로

  trans_net = torch.load(model_path)

  pred_list = real_img_test(net=trans_net, loader=real_dataloader)
  pred_ages = [label_to_age[pred] for pred in pred_list]
  ```
  
  ```
  # 이미지와 라벨 출력을 위한 함수 정의
  def plot_images_with_labels(image_paths, names, pred_ages):
      num_images = len(image_paths)
      fig, axs = plt.subplots(2, 4, figsize=(16, 8))

      plt.subplots_adjust(hspace=0.45)

      for i, image_path in enumerate(image_paths):
          image = Image.open(image_path)
          label = pred_ages[i]

          row = i // 4
          col = i % 4
          axs[row, col].imshow(image)
          axs[row, col].set_title(f"{label}", fontsize=10)
          axs[row, col].axis('off')

          label_x = image.size[0] / 2  # 이미지의 가로 중앙으로 텍스트를 이동시킵니다.
          label_y = image.size[1]  # 이미지의 아래에 텍스트를 표시합니다.
          axs[row, col].text(label_x, label_y+20, names[i], fontsize=10, ha='center', va='top')  # 이미지 밑에 라벨을 추가합니다.

      plt.show()

  # 이미지와 라벨을 그래프에 출력
  plot_images_with_labels(image_paths , names, pred_ages)
  ```
  <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/Korean_celebrity.png?raw=true">
  
- _The image above shows the age estimation result of a famous Korean celebrity image using TF-ResNet50._   
  <br/><br/><br/>
 
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

### 🚀 Machine Learning Model
<p>
  <img src="https://img.shields.io/badge/ResNet-2E8B57?style=flat-square?"/>
</p> 

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; AI Hub Dataset : Korean Face Image <br/>
