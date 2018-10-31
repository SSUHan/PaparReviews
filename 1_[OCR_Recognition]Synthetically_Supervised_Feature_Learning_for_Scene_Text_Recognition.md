# Synthetically Supervised Feature Learning for Scene Text Recognition

> ECCV 2018
>
> [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Liu_Synthetically_Supervised_Feature_ECCV_2018_paper.pdf)

### 0. Keywords

Scene Text Recognition, Deep Learning, Neural Networks, Feature Learning, Synthetic Data, Multitask Learning

Scene Text Recognition 필드에서 Synthetic Data 와 Multitask Learning 기법을 조합한 첫 시도로서의 의미를 갖는다.

### 1. Introduction

- Scene Text Recognition 이란 Natural scene image 로 부터 Text 를 인식시키는 문제를 의미한다
- 이는 다양한 산업환경에서 광범위하게 적용될 수 있기 때문에, 오래전부터 연구되어 왔다.
- Hand-crafted features 를 이용하는 방법부터 최근 SOTA 를 갱신하고 있는 Convolutional Neural Networks 까지 다양한 방법이 있는데, 
- 최근 SOTA 를 찍는 NN모델들의 핵심은 엄청난크기의(Large-scale) 합성이미지(Synthetic image dataset)를 학습에 사용하며(실제로 Recongition 학습때는 인식용데이터만 천만장이상 사용한다), 그 합성 데이터의 수준을 점점 높히며 모델 역시 함께 개선해가는 방법을 사용한다.
- 이는 OCR Recognition Task 는 합성데이터와 실제데이터의 간극이 크지 않기 때문이라는 특징이 있음

> The key idea of this work is that we can leverage the difference between real and synthetic images, namely the controllability of the generation process, and control the generation process to generate paired training data. Specifically, for every synthetic image out of the generation process with aforementioned nuisance fac- tors, we obtain the associated rendering parameters, manipulate the parameters, and generate a corresponding clean image where we remove part or all of the nuisance factors.

- 따라서 Training label 을 이용해, Nuisance real domain Image -> Clean synthetic domain image 로 페어를 만들 수가 있고, 우리는 이를 이용해 더 깨끗하고, Classification 에 더 도움이 되는 명확한 Feature 를 학습하게 하는 구조를 제안한다 고 주장한다.