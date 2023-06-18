# 👨‍👨‍👦‍👦🏻  Comparing the Effect of Transfer Learning on ResNet for Classification Problems
<br/>
  
### 1. &nbsp; Research Objective <br/><br/>

- _In this study, I train a ResNet50 model using the AI-Hub Korean Face Image dataset to estimate the age of individuals.  In the process, I compare the classification performance of the Original ResNet50 model and the TL-ResNet50 model that applies transfer learning (TL) technique. The main hypothesis of this research is as follows:_  <br/>

  - _"Under the condition of training with a small dataset and a small number of epochs, the model using transfer learning will perform better."_ <br/><br/>

- _This hypothesis is based on the following, referencing the ResNet-50 model structure from the paper “Deep Residual Learning for Image Recognition" (CVPR, 2016) and the transfer learning and fine-tuning experiments from the paper "Best Practices for Fine-tuning Visual Classifiers to New Domains" (ECCV, 2016)._  <br/>

  - _Model complexity and number of epochs: Overfitting may occur when training complex models with many layers and parameters (e.g., ResNet-50, VGG-19), which can be avoided by training with a small number of epochs. However, training with a small number of epochs may cause another problem: underfitting due to insufficient training on the entire dataset._ <br/>
  
  - _Data quantity: Training a model with a small dataset can lead to overfitting due to lack of diversity in the data. This can be addressed through transfer learning and fine-tuning, which leverages the generalization ability of pre-trained model trained with large datasets, and generally performs well even with a small amount of data (diversity) because it performs new tasks (model training) based on the weights of pre-trained model that have trained patterns from a variety of images._ <br/><br/>
    
- _The TL-ResNet50 model, which estimates the age of an individual based on the above hypothesis, is expected to be applied to various fields, such as predicting the age of a criminal._ <br/><br/><br/> 

### 2. &nbsp; Key Components of the Neural Network Model and Experimental Settings  <br/><br/>

- _Class labels_<br/>

  - _[0-6 years old,  7-12 years old, 13-19 years old, 20-30 years old, 31-45 years old, 46-55 years old, 56-66 years old, 67-80 years old]_ <br/><br/>
  
- _Number of Datasets_<br/>

  - _Training Dataset : 10025_<br/>
  - _Validation Dataset : 1539_<br/>
  - _Test Dataset : 1504_<br/><br/>

- _Model Architecture_ <br/>

    <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/ResNet_architecture.png?raw=true" width="400px">
    <br/><br/>
    
- _Batch Size : 512_  &nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp; _Learning Iterations : 0.01_ &nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;  _Optimization Algorithm : SGD(Stochastic Gradient Descent)_<br/><br/>
 
- _Loss Function : Cross-Entropy Loss Function_ &nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp; _Evaluation Metric : Accuracy_<br/><br/><br/> 

### 3. &nbsp; Data Preprocessing and Analysis <br/><br/>

- _**Package Settings**_ <br/> 
  
  ```
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  
  ```
  # colab에서 기본적으로 제공하지 않는 라이브러리 설치
  !pip install onnx
  ```
  
  ```
  import os
  import gc
  import time
  import onnx
  import zipfile
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim

  import torchvision
  import torchvision.models as models
  import torchvision.transforms as transforms

  from torch.utils.data import Dataset
  from torch.utils.data import DataLoader
  from torchsummary import summary

  from PIL import Image
  ```

- _**Data Preparation**_ <br/> 
  
  ```
  # 데이터셋 불러오기
  zip_path = '/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_korean_family_dataset_resolution_128.zip'
  output_dir = '/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_dataset'

  if os.path.exists(output_dir):
      print("이미 파일의 압축을 해제를 수행하였습니다.")
  else:
      print("파일의 압축을 해제를 수행합니다.")
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
          zip_ref.extractall(output_dir)
  ```
  
  ```
  """
  [Function] Parse the metadata.
  * image_age_list[0] = ["F0001_AGE_D_18_a1.jpg"] = "a"
  * image_age_list[1] = ["F0001_AGE_D_18_a2.jpg"] = "a"
  * image_age_list[2] = ["F0001_AGE_D_18_a3.jpg"] = "a"
  * image_age_list[3] = ["F0001_AGE_D_18_a4.jpg"] = "a"
  * image_age_list[4] = ["F0001_AGE_D_18_b1.jpg"] = "b"

  Training dataset: (F0001 ~ F0299) folders have 10,025 images.
  Validation dataset: (F0801 ~ F0850) folders have 1,539 images.
  Test dataset: (F0851 ~ F0900) folders have 1,504 images.'''
  """
  def parsing(meta_data):
      image_age_list = []

      # iterate all rows in the metadata file
      for idx, row in meta_data.iterrows():
          image_path = row['image_path']
          age_class = row['age_class']
          image_age_list.append([image_path, age_class])

      return image_age_list
  ```
  
  ```
  class Dataset(Dataset):
    def __init__(self, meta_data, image_directory, transform=None):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # process the meta data
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label
  ```
  
  ```
  train_meta_data_path = "/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_dataset/custom_train_dataset.csv"
  train_meta_data = pd.read_csv(train_meta_data_path)
  train_image_directory = "/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_dataset/train_images"

  val_meta_data_path = "/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_dataset/custom_val_dataset.csv"
  val_meta_data = pd.read_csv(val_meta_data_path)
  val_image_directory = "/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_dataset/val_images"

  test_meta_data_path = "/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_dataset/custom_test_dataset.csv"
  test_meta_data = pd.read_csv(test_meta_data_path)
  test_image_directory = "/content/drive/MyDrive/Colab Notebooks/Github_Repository/custom_dataset/test_images"

  '''
  [ 데이터 전처리 작업 정의 ]
  1) Resize() : 이미지의 크기를 128x128로 잘라내는 작업
  2) RandomHorizontalFlip() : 무작위로 이미지를 수평으로 뒤집는 작업 ==> 데이터 증강을 위해 사용되며, 이미지의 좌우 대칭을 통해 다양한 시각적 특성을 학습 가능
  3) ToTensor() : PIL 이미지 또는 NumPy 배열 형태의 이미지를 PyTorch 텐서로 변환
  4) Normalize() : 이미지의 각 채널을 평균과 표준편차를 사용하여 0과 1 사이의 범위로 변환 : 각 채널별 평균 값과 표준편차 값은 0.5로 설정
  '''

  train_transform = transforms.Compose([
      transforms.Resize(128),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  val_transform = transforms.Compose([
      transforms.Resize(128),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  test_transform = transforms.Compose([
      transforms.Resize(128),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  # 배치 사이즈 정의
  batch_size=512

  # 학습 Dataset 인스턴스 생성 후, DataLoader 인스턴스 정의
  train_dataset = Dataset(train_meta_data, train_image_directory, train_transform)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  # 검증 Dataset 인스턴스 생성 후, DataLoader 인스턴스 정의
  val_dataset = Dataset(val_meta_data, val_image_directory, val_transform)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  # 테스트 Dataset 인스턴스 생성 후, DataLoader 인스턴스 정의
  test_dataset = Dataset(test_meta_data, test_image_directory, test_transform)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  ```
  
  ```
  print(f"학습 데이터셋 개수 : {len(train_dataset)}개")
  print(f"검증 데이터셋 개수 : {len(val_dataset)}개")
  print(f"테스트 데이터셋 개수 : {len(test_dataset)}개")
  ```
  
  ```
  train_data_iter = iter(train_loader)
  train_input_data, train_labels = next(train_data_iter)

  val_data_iter = iter(val_loader)
  val_input_data, val_labels = next(val_data_iter)

  test_data_iter = iter(test_loader)
  test_input_data, test_labels = next(test_data_iter)

  print("학습 데이터의 입력 & 라벨 형상:", train_input_data.shape, '&', train_labels.shape)
  print("검증 데이터의 입력 & 라벨 형상:", val_input_data.shape, '&', val_labels.shape)
  print("테스트 데이터의 입력 & 라벨 형상:", test_input_data.shape, '&', test_labels.shape)
  ```
  
  ```
  # 8개의 이미지와 목표 변수(클래스)를 그래프로 출력
  label_to_age = {
      0: "0-6 years old",
      1: "7-12 years old",
      2: "13-19 years old",
      3: "20-30 years old",
      4: "31-45 years old",
      5: "46-55 years old",
      6: "56-66 years old",
      7: "67-80 years old"
  }

  plot_count = 0
  classes_seen = set()
  plt.figure(figsize=(24, 4))

  for i in range(len(train_dataset)):
      if plot_count == 8:
          break

      image, label = train_loader.dataset[i]
      image = np.transpose(image, (1, 2, 0))
      image = (image + 1) / 2  # -1~1 범위를 0~1 범위로 변환

      class_name = label_to_age[label]

      if class_name not in classes_seen:
          plt.subplot(1, 10, plot_count+1)
          plt.imshow(image)
          plt.title(class_name)
          plt.axis('off')

          classes_seen.add(class_name)
          plot_count += 1

  plt.show() # 그래프 출력
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/bar_graph1.png?raw=true">
  
- _**Exploratory Data Analysis (EDA)**_ <br/> 
  
  ```
  # 입력 데이터의 차원 변환 : 3차원(이미지 수, 3, 128, 128) -> 2차원 (이미지 수, 3*128*128)
  images = np.empty((len(train_dataset), 3, 128, 128), dtype=np.float32)
  labels = np.empty(len(train_dataset), dtype=np.int64)

  for i in range(len(train_dataset)):
      image, label = train_dataset[i]
      images[i] = image
      labels[i] = label

  x_train_reshaped = images.reshape(len(images), 128*128*3)

  # 데이터 프레임으로 변형하여 널 값의 빈도 확인
  x_train_df = pd.DataFrame(x_train_reshaped)
  total_null_count = x_train_df.isnull().sum().sum()

  print(f"널값의 개수 : {total_null_count}개")
  ```

  ```
  # 목표변수의 라벨별 빈도 계산 후 데이터 프레임으로 변환
  y_cnt = pd.DataFrame(labels).value_counts()
  df = pd.DataFrame(y_cnt, columns=['Count'])

  # 인덱스 리셋 및 문자열로 변환
  df.reset_index(inplace=True)
  df['Label'] = list(label_to_age.values())

  # 컬러맵 설정 및 바차트 생성
  cmap = plt.cm.Set3
  fig, ax = plt.subplots(figsize=(16, 3))
  bars = ax.bar(df['Label'], df['Count'], color=cmap(np.arange(len(df))))

  # 바 위에 라벨 갯수 출력
  for i, count in enumerate(df['Count']):
      ax.text(i, count + 100, str(count), ha='center', fontsize=7)

  # 그래프 레이블과 제목 설정 및  y축 범위 늘리기 (현재 최댓값의 110%로 범위 지정)
  ax.set_xlabel('Label')
  ax.set_ylabel('Frequency')
  ax.set_title('Label Counts')
  ax.set_ylim(0, df['Count'].max() * 1.1)

  plt.show() # 그래프 출력
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/img_graph.png?raw=true">

### 3. &nbsp; Training and Testing ResNet50 Model <br/><br/>

- _**Defining the ResNet50 Model Architecture**_ <br/> 

  ```
  def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
      r"""
      3x3 convolution with padding
      - in_planes: in_channels
      - out_channels: out_channels
      - bias=False: BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정.
      """
      return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, bias=False, dilation=dilation)

  def conv1x1(in_planes, out_planes, stride=1):
      """1x1 convolution"""
      return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

  class Bottleneck(nn.Module):
      expansion = 4 # 블록 내에서 차원을 증가시키는 3번째 conv layer에서의 확장계수

      def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
          super(Bottleneck, self).__init__()

          if norm_layer is None:
              norm_layer = nn.BatchNorm2d

          # ResNext나 WideResNet의 경우 사용
          width = int(planes * (base_width / 64.)) * groups

          # Bottleneck Block의 구조
          self.conv1 = conv1x1(inplanes, width)
          self.bn1 = norm_layer(width)
          # conv2에서 downsample
          self.conv2 = conv3x3(width, width, stride, groups, dilation)
          self.bn2 = norm_layer(width)
          self.conv3 = conv1x1(width, planes * self.expansion)
          self.bn3 = norm_layer(planes * self.expansion)
          self.relu = nn.ReLU(inplace=True)
          self.downsample = downsample
          self.stride = stride

      def forward(self, x):
          identity = x
          # 1x1 convolution layer
          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)
          # 3x3 convolution layer
          out = self.conv2(out)
          out = self.bn2(out)
          out = self.relu(out)
          # 1x1 convolution layer
          out = self.conv3(out)
          out = self.bn3(out)

          # skip connection
          if self.downsample is not None:
              identity = self.downsample(x)

          out += identity
          out = self.relu(out)

          return out

  class ResNet50(nn.Module):
      def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
          super(ResNet50, self).__init__()

          if norm_layer is None:
              norm_layer = nn.BatchNorm2d

          self._norm_layer = norm_layer
          # input feature map
          self.inplanes = 64
          self.dilation = 1

          # stride를 dilation으로 대체할지 선택
          if replace_stride_with_dilation is None:
              replace_stride_with_dilation = [False, False, False]

          if len(replace_stride_with_dilation) != 3:
              raise ValueError("replace_stride_with_dilation should be None "
                               "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

          self.groups = groups
          self.base_width = width_per_group

          r"""
          - 처음 입력에 적용되는 self.conv1과 self.bn1, self.relu는 모든 ResNet에서 동일
          - 3: 입력으로 RGB 이미지를 사용하기 때문에 convolution layer에 들어오는 input의 channel 수는 3
          """
          self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
          self.bn1 = norm_layer(self.inplanes)
          self.relu = nn.ReLU(inplace=True)
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

          r"""
          - 아래부터 block 형태와 갯수가 ResNet층마다 변화
          - self.layer1 ~ 4: 필터의 개수는 각 block들을 거치면서 증가(64->128->256->512)
          - self.avgpool: 모든 block을 거친 후에는 Adaptive AvgPool2d를 적용하여 (n, 512, 1, 1)의 텐서로
          - self.fc: 이후 fc layer를 연결
          """
          self.layer1 = self._make_layer(block, 64, layers[0])
          self.layer2 = self._make_layer(block, 128, layers[1], stride=2, # 여기서부터 downsampling적용
                                         dilate=replace_stride_with_dilation[0])
          self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                         dilate=replace_stride_with_dilation[1])
          self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                         dilate=replace_stride_with_dilation[2])
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
          self.fc = nn.Linear(512 * block.expansion, num_classes)

          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)

          # Zero-initialize the last BN in each residual branch,
          # so that the residual branch starts with zeros, and each residual block behaves like an identity.
          if zero_init_residual:
              for m in self.modules():
                  if isinstance(m, Bottleneck):
                      nn.init.constant_(m.bn3.weight, 0)

      def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
          r"""
          convolution layer 생성 함수
          - block: block종류 지정
          - planes: feature map size (input shape)
          - blocks: layers[0]와 같이, 해당 블록이 몇개 생성돼야하는지, 블록의 갯수 (layer 반복해서 쌓는 개수)
          - stride와 dilate은 고정
          """
          norm_layer = self._norm_layer
          downsample = None
          previous_dilation = self.dilation

          if dilate:
              self.dilation *= stride
              stride = 1

          # the number of filters is doubled: self.inplanes와 planes 사이즈를 맞춰주기 위한 projection shortcut
          # the feature map size is halved: stride=2로 downsampling
          if stride != 1 or self.inplanes != planes * block.expansion:
              downsample = nn.Sequential(
                  conv1x1(self.inplanes, planes * block.expansion, stride),
                  norm_layer(planes * block.expansion),
              )

          layers = []

          # 블록 내 시작 layer, downsampling 필요
          layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                              self.base_width, previous_dilation, norm_layer))
          self.inplanes = planes * block.expansion # inplanes 업데이트

          # 동일 블록 반복
          for _ in range(1, blocks):
              layers.append(block(self.inplanes, planes, groups=self.groups,
                                  base_width=self.base_width, dilation=self.dilation,
                                  norm_layer=norm_layer))

          return nn.Sequential(*layers)

      def _forward_impl(self, x):
          x = self.conv1(x)
          x = self.bn1(x)
          x = self.relu(x)
          x = self.maxpool(x)

          x = self.layer1(x)
          x = self.layer2(x)
          x = self.layer3(x)
          x = self.layer4(x)

          x = self.avgpool(x)
          x = torch.flatten(x, 1)
          x = self.fc(x)

          return x

      def forward(self, x):
          return self._forward_impl(x)
  ```

  ```
  # ResNet 50 모델 구조 출력
  dummy_model = ResNet50(Bottleneck, [3, 4, 6, 3]).cuda()
  summary(dummy_model, input_size = (3 , 128, 128))
  ```
    
  ```
  # ResNet50 모델 구조 onnx 파일로 저장
  path = 'dummy_model_output.onnx'
  dummy_model = ResNet50(Bottleneck, [3, 4, 6, 3]).cuda()
  dummy_data = torch.empty(1, 3 , 128, 128, dtype=torch.float32).cuda()

  torch.onnx.export(dummy_model, dummy_data, path)
  onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
  ```
  <br/> 
### 3. &nbsp; Training and Testing ResNet50 Model <br/><br/>

- _**Apply Transfer Learning and Define the "Train / Validate / Test" Function**_ <br/> 

  ```
  learning_rate = 0.01
  log_step = 1

  # 사전 학습된 ResNet50을 가져옴
  pretrained_model = models.resnet50(pretrained=True)
  num_features = pretrained_model.fc.in_features

  org_net = ResNet50(Bottleneck, [3, 4, 6, 3])
  org_net.fc = nn.Linear(num_features, 8)
  org_net = org_net.to('cuda')
  org_criterion = nn.CrossEntropyLoss()
  org_optimizer = optim.SGD(org_net.parameters(), lr=learning_rate, momentum=0.9)

  trans_net = ResNet50(Bottleneck, [3, 4, 6, 3])
  trans_net.load_state_dict(pretrained_model.state_dict()) # transfer learning
  trans_net.fc = nn.Linear(num_features, 8)
  trans_net = trans_net.to('cuda')
  trans_criterion = nn.CrossEntropyLoss()
  trans_optimizer = optim.SGD(trans_net.parameters(), lr=learning_rate, momentum=0.9)

  def train(net, optimizer, criterion, epoch):
      print(f'[Epoch: {epoch + 1} - Training]')

      net.train()

      total = 0
      running_loss = 0.0
      running_corrects = 0

      for i, batch in enumerate(train_loader):
          imgs, labels = batch
          imgs, labels = imgs.to("cuda"), labels.to("cuda")

          optimizer.zero_grad()

          outputs = net(imgs)
          _, preds = torch.max(outputs, 1)

          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          total += labels.shape[0]
          running_loss += loss.item()
          running_corrects += torch.sum(preds == labels.data)

          if i % log_step == log_step - 1:
              print(f'\t\t[Batch: {i + 1} / {len(train_loader)}] running train loss: {running_loss / total}, running train accuracy: {running_corrects / total}')

      return running_loss / total, (running_corrects / total).item()

  def validate(net, criterion, epoch):
      print(f'[Epoch: {epoch + 1} - Validation]')

      net.eval()

      total = 0
      running_loss = 0.0
      running_corrects = 0

      for i, batch in enumerate(val_loader):
          imgs, labels = batch
          imgs, labels = imgs.cuda(), labels.cuda()

          with torch.no_grad():
              outputs = net(imgs)
              _, preds = torch.max(outputs, 1)
              loss = criterion(outputs, labels)

          total += labels.shape[0]
          running_loss += loss.item()
          running_corrects += torch.sum(preds == labels.data)

          if (i == 0) or (i % log_step == log_step - 1):
              print(f'\t\t[Batch: {i + 1} / {len(val_loader)}] running val loss: {running_loss / total}, running val accuracy: {running_corrects / total}')
      print()

      return running_loss / total, (running_corrects / total).item()

  def test(net, criterion):
      net.eval()

      total = 0
      running_loss = 0.0
      running_corrects = 0

      for i, batch in enumerate(test_loader):
          imgs, labels = batch
          imgs, labels = imgs.cuda(), labels.cuda()

          with torch.no_grad():
              outputs = net(imgs)
              _, preds = torch.max(outputs, 1)
              loss = criterion(outputs, labels)

          total += labels.shape[0]
          running_loss += loss.item()
          running_corrects += torch.sum(preds == labels.data)

      return running_loss / total, (running_corrects / total).item()
  ```
- _**Training & Testing the Original-ResNet50 Model**_ <br/> 

  ```
  gc.collect()
  torch.cuda.empty_cache()
  org_train_loss, org_train_acc, org_val_loss, org_val_acc = [], [], [], []

  for epoch in range(0, 10):
      train_loss, train_acc = train(net=org_net, optimizer=org_optimizer, criterion=org_criterion, epoch=epoch)
      val_loss, val_acc = validate(net=org_net, criterion=org_criterion, epoch=epoch)

      org_train_loss.append(train_loss)
      org_train_acc.append(train_acc)
      org_val_loss.append(val_loss)
      org_val_acc.append(val_acc)
  ```
- _**Training & Testing the TL-ResNet50 Model**_ <br/>   
  ```
  gc.collect()
  torch.cuda.empty_cache()
  trans_train_loss, trans_train_acc, trans_val_loss, trans_val_acc = [], [], [], []

  for epoch in range(0, 10):
      train_loss, train_acc = train(net=trans_net, optimizer=trans_optimizer, criterion=trans_criterion, epoch=epoch)
      val_loss, val_acc = validate(net=trans_net, criterion=trans_criterion, epoch=epoch)

      trans_train_loss.append(train_loss)
      trans_train_acc.append(train_acc)
      trans_val_loss.append(val_loss)
      trans_val_acc.append(val_acc)
  ```
  <br/> 

### 4. &nbsp; Research Results  <br/><br/>
    
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
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Transfer-Learning-on-ResNet-for-Classification-Problems/blob/main/image/bar_graph2.png?raw=true" alt="bar_graph2" width="640" >&nbsp;&nbsp;&nbsp;&nbsp;
  
- _Finally, I compared the accuracy values of the two models on real test data and found that the TF-ResNet model has about 2x higher accuracy than the Original-ResNet50 model. These results prove that models using transfer learning perform better when training with small datasets and a limited number of epochs._ <br/> <br/> <br/>
 
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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; AI Hub Dataset : Korean Face Image Dataset <br/>
