# **CycleGAN-VC2-PyTorch**

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/jackaduma/CycleGAN-VC2)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/jackaduma?locale.x=zh_XC)

[**中文说明**](./README.zh-CN.md) | [**English**](./README.md)

------

This code is a **PyTorch** implementation for paper: [CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion](https://arxiv.org/abs/1904.04631]), a nice work on **Voice-Conversion/Voice Cloning**.

- [x] Dataset
  - [ ] VC
  - [x] Chinese Male Speakers (S0913 from [AISHELL-Speech](https://openslr.org/33/) & [GaoXiaoSong: a Chinese star](https://en.wikipedia.org/wiki/Gao_Xiaosong))
- [x] Usage
  - [x] Training
  - [x] Example 
- [ ] Demo
- [x] Reference

------

## **Update**

**2020.11.17**: fixed issues: re-implements the second step adverserial loss.

**2020.08.27**: add the second step adverserial loss by [Jeffery-zhang-nfls](https://github.com/Jeffery-zhang-nfls)

## **CycleGAN-VC2**

### [**Project Page**](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)


To advance the research on non-parallel VC, we propose CycleGAN-VC2, which is an improved version of CycleGAN-VC incorporating three new techniques: an improved objective (two-step adversarial losses), improved generator (2-1-2D CNN), and improved discriminator (Patch GAN).


![network](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/images/network.png "network")

------

**This repository contains:** 

1. [model code](model_tf.py) which implemented the paper.
2. [audio preprocessing script](preprocess_training.py) you can use to create cache for [training data](data).
3. [training scripts](train.py) to train the model.
4. [Examples of Voice Conversion](converted_sound/) - converted result after training.

------

## **Table of Contents**

- [**CycleGAN-VC2-PyTorch**](#cyclegan-vc2-pytorch)
  - [**Update**](#update)
  - [**CycleGAN-VC2**](#cyclegan-vc2)
    - [**Project Page**](#project-page)
  - [**Table of Contents**](#table-of-contents)
  - [**Requirement**](#requirement)
  - [**Usage**](#usage)
    - [**preprocess**](#preprocess)
    - [**train**](#train)
  - [**Pretrained**](#pretrained)
  - [**Demo**](#demo)
  - [**Star-History**](#star-history)
  - [**Reference**](#reference)
  - [Donation](#donation)
  - [**License**](#license)
  
------



## **Requirement** 

```bash
pip install -r requirements.txt
```
## **Usage**

### **preprocess**

```python
python preprocess_training.py
```
is short for

```python
python preprocess_training.py --train_A_dir ./data/S0913/ --train_B_dir ./data/gaoxiaosong/ --cache_folder ./cache/
```


### **train** 
```python
python train.py
```

is short for

```python
python train.py --logf0s_normalization ./cache/logf0s_normalization.npz --mcep_normalization ./cache/mcep_normalization.npz --coded_sps_A_norm ./cache/coded_sps_A_norm.pickle --coded_sps_B_norm ./cache/coded_sps_B_norm.pickle --model_checkpoint ./model_checkpoint/ --resume_training_at ./model_checkpoint/_CycleGAN_CheckPoint --validation_A_dir ./data/S0913/ --output_A_dir ./converted_sound/S0913 --validation_B_dir ./data/gaoxiaosong/ --output_B_dir ./converted_sound/gaoxiaosong/
```

------

## **Pretrained**

a pretrained model which converted between S0913 and GaoXiaoSong

download from [Google Drive](https://drive.google.com/file/d/1iamizL98NWIPw4pw0nF-7b6eoBJrxEfj/view?usp=sharing) <735MB>

------

## **Demo**

Samples:


**reference speaker A:** [S0913(./data/S0913/BAC009S0913W0351.wav)](https://drive.google.com/file/d/14zU1mI8QtoBwb8cHkNdZiPmXI6Mj6pVW/view?usp=sharing)

**reference speaker B:** [GaoXiaoSong(./data/gaoxiaosong/gaoxiaosong_1.wav)](https://drive.google.com/file/d/1s0ip6JwnWmYoWFcEQBwVIIdHJSqPThR3/view?usp=sharing)



**speaker A's speech changes to speaker B's voice:** [Converted from S0913 to GaoXiaoSong (./converted_sound/S0913/BAC009S0913W0351.wav)](https://drive.google.com/file/d/1S4vSNGM-T0RTo_aclxRgIPkUJ7NEqmjU/view?usp=sharing)

------
## **Star-History**

![star-history](https://api.star-history.com/svg?repos=jackaduma/CycleGAN-VC2&type=Date "star-history")

------

## **Reference**
1. **CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion**. [Paper](https://arxiv.org/abs/1904.04631), [Project](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)
2. Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks. [Paper](https://arxiv.org/abs/1711.11293), [Project](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/)
3. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. [Paper](https://arxiv.org/abs/1703.10593), [Project](https://junyanz.github.io/CycleGAN/), [Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
4. Image-to-Image Translation with Conditional Adversarial Nets. [Paper](https://arxiv.org/abs/1611.07004), [Project](https://phillipi.github.io/pix2pix/), [Code](https://github.com/phillipi/pix2pix)

------

## Donation
If this project help you reduce time to develop, you can give me a cup of coffee :) 

AliPay(支付宝)
<div align="center">
	<img src="./misc/ali_pay.png" alt="ali_pay" width="400" />
</div>

WechatPay(微信)
<div align="center">
    <img src="./misc/wechat_pay.png" alt="wechat_pay" width="400" />
</div>

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://paypal.me/jackaduma?locale.x=zh_XC)

------

## **License**

[MIT](LICENSE) © Kun




(1)target 

Voice conversion method using improved cycleGAN based on parallel data called cycleGAN-vc2 which has three methods: Improved Objective:
Two-Step Adversarial Losses, Improved Generator: 2-1-2D CNN,Improved Discriminator: PatchGAN These , these three methods are the same improved methods in
cycleGAN-vc and they were : Objective: One-Step Adversarial Loss,Generator: 1D CNN, Discriminator: FullGAN and for explain the codes This code is a 
PyTorch implementation and the libraris that we used in codes were :torch.nn as nn,torch, numpy as np, etc.
and some classes like: GLU, PixelShuffle,ResidualLayer,....
 downSample_Generator,Generator

(2)explain the innovation 

my innovation was about combine the cycleGAN-vc2 with cycleGAN-vc with new method sucj as Objective Evaluation, and Subjective Evaluation that  results show that CycleGAN-VC2 outperforms
CycleGAN-VC in terms of both naturalness and similarity for every speaker pair. Particularly, CycleGAN-VC is difficult
to apply to a vocoder-free VC framework(used in SF-TF and SM-TM), as this framework is sensitive to conversion error due to the usage of differential MCEPs. However,
the MOS indicates that CycleGAN-VC2 works relatively. well in such a difficult setting. in addition use a new method as sTARGAN for this innovation we have to use more libraries
in our codes,such as:torch,os,random,data,transform, tensorflow as tf,scipy.misc . And we also defined newer classes like : CelebA, Solver, ResidualBlock, 
 Discriminator. in STARGAN method at first we set a Model and then describe model configurations and with these codes We make changes to the model also
 by the cycleGAN-VC2  we can define some layers and change them by their self, in_channels, out_channels, kernel_size, stride, padding)


(3)The source code written in the data set about cycleGAN-vc2

#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 14:36

from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class trainingDataset(Dataset):
    def __init__(self, datasetA, datasetB, n_frames=128):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n_frames = n_frames

    def __getitem__(self, index):
        dataset_A = self.datasetA
        dataset_B = self.datasetB
        n_frames = self.n_frames

        self.length = min(len(dataset_A), len(dataset_B))

        num_samples = min(len(dataset_A), len(dataset_B))
        
        
        train_data_A_idx = np.arange(len(dataset_A))
        train_data_B_idx = np.arange(len(dataset_B))
        np.random.shuffle(train_data_A_idx)
        np.random.shuffle(train_data_B_idx)
        train_data_A_idx_subset = train_data_A_idx[:num_samples]
        train_data_B_idx_subset = train_data_B_idx[:num_samples]

        train_data_A = list()
        train_data_B = list()

        for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
            data_A = dataset_A[idx_A]
            frames_A_total = data_A.shape[1]
            assert frames_A_total >= n_frames
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            train_data_A.append(data_A[:, start_A:end_A])

            data_B = dataset_B[idx_B]
            frames_B_total = data_B.shape[1]
            assert frames_B_total >= n_frames
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            train_data_B.append(data_B[:, start_B:end_B])

        train_data_A = np.array(train_data_A)
        train_data_B = np.array(train_data_B)

        return train_data_A[index], train_data_B[index]

    def __len__(self):
        return min(len(self.datasetA), len(self.datasetB))


# if __name__ == '__main__':
#     trainA = np.random.randn(162, 24, 554)
#     trainB = np.random.randn(158, 24, 554)
#     dataset = trainingDataset(trainA, trainB)
#     trainLoader = torch.utils.data.DataLoader(dataset=dataset,
#                                               batch_size=2,
#                                               shuffle=True)
#     for epoch in range(10):
#         for i, (trainA, trainB) in enumerate(trainLoader):
#             print(trainA.shape, trainB.shape)

 data base linke 
 https://arxiv.org/abs/1904.04631%5D
 https://openslr.org/33/
 https://en.wikipedia.org/wiki/Gao_Xiaosong
 
 the files of data base 
 https://github.com/jackaduma/CycleGAN-VC2/tree/master/data/gaoxiaosong
 
 change sorce code
 
import librosa
import os
import pyworld
from pprint import pprint
import time


and 

new code 

import os
import tensorflow as tf
from module import * #discriminator, generator_gatedcnn
from utils import l1_loss, l2_loss, cross_entropy_loss
from datetime import datetime

class CycleGAN2(object):

    def __init__(self, num_features, mode = 'train',
                 log_dir = './log', model_name='tmp.ckpt'):

        self.num_features = num_features
        self.input_shape = [None, num_features, None] # [batch_size, num_features, num_frames]

        self.generator = generator_gated2Dcnn
        self.discriminator = discriminator_2D
        
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            self.log_dir = os.path.join(log_dir, os.path.splitext(model_name)[0])
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):

        # ---------------- Placeholders for real training samples ---------------- #
        self.input_A_real = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_B_real')
        # Placeholders for fake generated samples
        self.input_A_fake = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_A_fake')
        self.input_B_fake = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_B_fake')
        # Placeholder for test samples
        self.input_A_test = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_A_test')
        self.input_B_test = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_B_test')
        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.placeholder(tf.float32, None, name='lambda_cycle')
        self.lambda_identity = tf.placeholder(tf.float32, None, name='lambda_identity')
        self.lambda_feat = tf.placeholder(tf.float32, None, name='lambda_identity')
        # ---------------- Placeholders for real training samples ---------------- #

        # ---------------- CycleGAN, Generator ---------------- #
        # Generator A->AtoB->AtoBtoA, input_A_real->generation_B->cycle_A
        self.generation_B = self.generator(inputs=self.input_A_real, reuse=False, scope_name='generator_A2B')
        self.cycle_A = self.generator(inputs=self.generation_B, reuse=False, scope_name='generator_B2A')

        # Generator B->BtoA->BtoAtoB, input_B_real->generation_A->cycle_B
        self.generation_A = self.generator(inputs=self.input_B_real, reuse=True, scope_name='generator_B2A')
        self.cycle_B = self.generator(inputs=self.generation_A, reuse=True, scope_name='generator_A2B')
        # ---------------- CycleGAN, Generator ---------------- #

        # ---------------- Identity/Feature loss ---------------- #
        self.generation_A_identity = self.generator(inputs=self.input_A_real, reuse=True, scope_name='generator_B2A')
        self.generation_B_identity = self.generator(inputs=self.input_B_real, reuse=True, scope_name='generator_A2B')
        # ---------------- Identity/Feature loss ---------------- #

        # ---------------- CycleGAN, Discriminator ---------------- #
        self.discrimination_A_fake = self.discriminator(inputs = self.generation_A, reuse = False, scope_name = 'discriminator_A')
        self.discrimination_B_fake = self.discriminator(inputs = self.generation_B, reuse = False, scope_name = 'discriminator_B')
        # ---------------- CycleGAN, Discriminator ---------------- #

        # ---------------- Loss Define ---------------- #
        # Cycle loss
        self.cycle_loss = l1_loss(y = self.input_A_real, y_hat = self.cycle_A) + l1_loss(y = self.input_B_real, y_hat = self.cycle_B)

        # Identity loss
        self.identity_loss = l1_loss(y = self.input_A_real, y_hat = self.generation_A_identity) + l1_loss(y = self.input_B_real, y_hat = self.generation_B_identity)
        # ---------------- Loss Define ---------------- #

        # ---------------- Loss Calculation ---------------- #
        # ================ Generator loss ================ #
        # Generator wants to fool discriminator
        self.generator_loss_A2B = l2_loss(y = tf.ones_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)
        self.generator_loss_B2A = l2_loss(y = tf.ones_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)

        # Merge the two generators and the cycle loss
        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + self.lambda_cycle * self.cycle_loss + self.lambda_identity * self.identity_loss
        # ================ Generator loss ================ #
        # ================ Discriminator loss ================ #
        self.discrimination_input_A_real = self.discriminator(inputs = self.input_A_real, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_real = self.discriminator(inputs = self.input_B_real, reuse = True, scope_name = 'discriminator_B')
        self.discrimination_input_A_fake = self.discriminator(inputs = self.input_A_fake, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_fake = self.discriminator(inputs = self.input_B_fake, reuse = True, scope_name = 'discriminator_B')

        # Discriminator wants to classify real and fake correctly
        self.discriminator_loss_input_A_real = l2_loss(y = tf.ones_like(self.discrimination_input_A_real), y_hat = self.discrimination_input_A_real)
        self.discriminator_loss_input_A_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_A_fake), y_hat = self.discrimination_input_A_fake)
        self.discriminator_loss_A = (self.discriminator_loss_input_A_real + self.discriminator_loss_input_A_fake) / 2

        self.discriminator_loss_input_B_real = l2_loss(y = tf.ones_like(self.discrimination_input_B_real), y_hat = self.discrimination_input_B_real)
        self.discriminator_loss_input_B_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_B_fake), y_hat = self.discrimination_input_B_fake)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake) / 2

        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B
        # ================ Discriminator loss ================ #
        # ---------------- Loss Calculation ---------------- #

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]

        # ---------------- Reserved for test ---------------- #
        self.generation_B_test = self.generator(inputs=self.input_A_test, reuse=True, scope_name='generator_A2B')
        self.generation_A_test = self.generator(inputs=self.input_B_test, reuse=True, scope_name='generator_B2A')
        # ---------------- Reserved for test ---------------- #

    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, None, name = 'generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name = 'discriminator_learning_rate')
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = self.discriminator_learning_rate, beta1 = 0.5).minimize(self.discriminator_loss, var_list = self.discriminator_vars)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate = self.generator_learning_rate, beta1 = 0.5).minimize(self.generator_loss, var_list = self.generator_vars) 

    def train(self, input_A, input_B, lambda_cycle, lambda_identity, generator_learning_rate, discriminator_learning_rate):

        generation_A, generation_B, generator_loss, _, generator_summaries, generator_loss_A2B = self.sess.run(
            [self.generation_A, self.generation_B, self.generator_loss, self.generator_optimizer, self.generator_summaries, self.generator_loss_A2B], \
            feed_dict = {self.lambda_cycle: lambda_cycle, self.lambda_identity: lambda_identity, self.input_A_real: input_A, self.input_B_real: input_B, self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run([self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries], \
            feed_dict = {self.input_A_real: input_A, self.input_B_real: input_B, self.discriminator_learning_rate: discriminator_learning_rate, self.input_A_fake: generation_A, self.input_B_fake: generation_B})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return generator_loss, discriminator_loss, generator_loss_A2B


    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generation_B_test, feed_dict = {self.input_A_test: inputs})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict = {self.input_B_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge([cycle_loss_summary, identity_loss_summary, generator_loss_A2B_summary, generator_loss_B2A_summary, generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_A_summary, discriminator_loss_B_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    
    model = CycleGAN2(num_features = 24)
    print('Graph Compile Successeded.')
    
    
    https://github.com/alpharol/Voice_Conversion_CycleGAN2/blob/master/model.py
    erja be in proje
     
(4)the result of new code 
    
    
The filters of the 1D -> 2D of Generator is decreased from 2304 to 512, and the filters of the last conv in the Generator is decreased from 35 to 1.
it may take 2 minutes for one epoch. 500 epoch are needed in order to get good voice quality.
The pre-process is separated from the educational stages. Added process bar to the code. Training speed has been increased by saving models up to 100 courses.
The last five courses storage module has been added.Data save as HDF5 format 
(world_decompose extracts f0, aperiodicity and spectral envelope. This function is computationally intensive.)


(5) refrence 
https://github.com/onejiin/CycleGAN-VC2


(6)Introduction my self
homa sepehr ghazvini nejad . born on 1377, tehran city , terme 3 master  biomedical engineering , tehran jonob university 


(7)
https://docs.google.com/document/d/17L1ojBzlhMxfC63IUqtJgrTduLdJe3Mw/edit?usp=sharing&ouid=109569480816806143532&rtpof=true&sd=true
https://docs.google.com/document/d/17L1ojBzlhMxfC63IUqtJgrTduLdJe3Mw/edit?usp=sharing&ouid=109569480816806143532&rtpof=true&sd=true


(8) video of explainning of codes and articles 

https://drive.google.com/file/d/1CUvTbl-1VCQ9I5b0QECePHv3HSTiT_K1/view?usp=drivesdk
https://drive.google.com/file/d/100P_YlkqsAjnLRFTqH9IefaOt91RCfIM/view?usp=drivesdk
https://drive.google.com/file/d/1Z1qCF5GCe9LWlyYssW5Z1NiLAEPbHjyZ/view?usp=sharing
https://drive.google.com/file/d/1y273Ncb1oFLz8t2f4_XNxxY_PpfGbn2P/view?usp=share_link
https://drive.google.com/file/d/1duq6bdvIyFPHCozJY0Hf6ouRuLSkI8rD/view?usp=drivesdk

(9)

https://docs.google.com/document/d/1Db9PH4i5TF4WUBfT3dApQgCxEZK6IrAO/edit?usp=sharing&ouid=109569480816806143532&rtpof=true&sd=true

