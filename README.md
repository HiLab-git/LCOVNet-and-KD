# Efficient Multi-Organ Segmentation from 3D Abdominal CT Images with Lightweight Network and Knowledge Distillation
[tmi_link]:https://ieeexplore.ieee.org/document/10083150
[isbi_link]:https://ieeexplore.ieee.org/abstract/document/9434023
[word_link]:https://www.sciencedirect.com/science/article/abs/pii/S1361841522002705
[pymic_link]:https://github.com/HiLab-git/PyMIC
[pymic_example]:https://github.com/HiLab-git/PyMIC_examples
[baidu_link]:https://pan.baidu.com/s/1HwD1iqHorgXfYXnrChdzIg
This repository provides the code for "Efficient Multi-Organ Segmentation from 3D Abdominal CT Images with Lightweight Network and Knowledge Distillation"(accepted by [TMI][tmi_link]) and "https://ieeexplore.ieee.org/abstract/document/9434023"(published by [ISBI][isbi_link]).

![result](./pic/result.png)
Visual comparison between different networks for abdominal organ segmentation on the [WORD][word_link] dataset.

![structure](./pic/kd_structure.png)
Overview of our proposed lightweight LCOV-Net and KD strategies. LCOV-Net is built on our Lightweight Attention-based Convolutional Blocks (LACB-H and LACB-L) to reduce the model size. To improve itsmperformance, we introduce Class-Affinity Knowledge Distillation (CAKD) and Multi-Scale Knowledge Distillation (MSKD) as shown in (c) to effectively distill knowledge from a heavy-weight teacher model to LCOV-Net. Note that for simplicity, the KD losses are only shown for the highest resolution level.

![structure](./pic/lcovnet_structure.png)
Our proposed LACB for efficient computation.


# DataSet
Please contact Xiangde (luoxd1996 AT gmail DOT com) for the dataset (**the label of the testing set can be downloaded now [labelTs](https://github.com/HiLab-git/WORD/blob/main/WORD_V0.1.0_labelsTs.zip)**). Two steps are needed to download and access the dataset: **1) using your google email to apply for the download permission ([Goole Driven](https://drive.google.com/drive/folders/16qwlCxH7XtJD9MyPnAbmY4ATxu2mKu67?usp=sharing), [BaiduPan](https://pan.baidu.com/s/1mXUDbUPgKRm_yueXT6E_Kw))**; **2) using your affiliation email to get the unzip password/BaiduPan access code**. We will get back to you within **two days**, **so please don't send them multiple times**. We just handle the **real-name email** and **your email suffix must match your affiliation**. The email should contain the following information:

    Name/Homepage/Google Scholar: (Tell us who you are.)
    Primary Affiliation: (The name of your institution or university, etc.)
    Job Title: (E.g., Professor, Associate Professor, Ph.D., etc.)
    Affiliation Email: (the password will be sent to this email, we just reply to the email which is the end of "edu".)
    How to use: (Only for academic research, not for commercial use or second-development.)
    
In addition, this work is still ongoing, the **WORD** dataset will be extended to larger and more diverse (more patients, more organs, and more modalities, more clinical hospitals' data and MR Images will be considered to include future), any **suggestion**, **comment**, **collaboration**, and **sponsor** are welcome. 

# How to use
1. Download the pretrained model and example CT images from [Baidu Netdisk][baidu_link] (extract code 9jlj).
2. Run `./KD/run.sh`. The results will be saved in `./KD/model/kd`.

# How to train COPLE-Net
Training was implemented with [PyMIC][pymic_link].

Just follow these [examples][pymic_example] for using PyMIC for network training and testing.

You may need to custormize the configure files to use different network structures, preprocessing methods and loss functions.