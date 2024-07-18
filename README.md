# 1. TSDM-Fusion
W. Xue, Y. Liu, F. Wang, G. He and Y. Zhuang, "A Novel Teacher–Student Framework With Degradation Model for Infrared–Visible Image Fusion," in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-12, 2024, Art no. 5020512, doi: 10.1109/TIM.2024.3398115.

**paper in**

https://doi.org/10.1109/TIM.2024.3398115.

https://ieeexplore.ieee.org/document/10522962.

# 2. Abstract:
The fusion performance of infrared (IR) and visible (VIS) images depends on the quality of the source images, which are often affected by some factors in real-world scenarios, such as environmental changes, hardware limitations, and image compression. The influence of these factors can be minimized by training a neural network capable of generating high-quality (HQ) fused images from low-quality (LQ) source images. However, in real-world conditions, it is challenging to acquire paired LQ source images and their corresponding HQ fused images for network training. To address this issue, we propose a novel teacher–student framework with a degradation model for IR–VIS image fusion, namely, TSDM-Fusion. In this framework, the teacher network is utilized to generate HQ RGB fused images, while the degradation model is employed to generate LQ IR and VIS images. Subsequently, the obtained images are used to train the student network, enabling it to learn the mapping from LQ IR and RGB VIS images to HQ RGB fused images. The design for the degradation model is the most important part of the framework, which simulates degradation factors in the imaging processes, including brightness, contrast, blur, noise, and JPEG compression. Experiments on multiple public image fusion datasets and M3FD detection datasets demonstrate that our method can not only enhance the visual fusion effects but also improve the detection mAP. The code and pretrained models are available at https://github.com/bearxwm/TSDM-Fusion.

# 3. Visual Compare Results
## 3.1 TNO
TNO: Visual comparison results on the TNO dataset. (a) and (b) are infrared and visible images, while (c)-(i) are the results of image fusion algorithms and (j) is the result of our method. Please zoom in on the images to see more details. ![](./ComparedImages/TNO.png)
## 3.2 ROAD SCENE
ROAD SCENE: Visual comparison results on the Road Scene dataset. (a) and (b) are infrared and visible images, while (c)-(i) are the results of image fusion algorithms and (j) is the result of our method. Please zoom in on the images to see more details. ![](./ComparedImages/ROAD.png)
## 3.3 LLVIP
LLVIP: Visual comparison results on the LLVIP dataset. (a) and (b) are infrared and visible images, (c)-(i) are the results of image fusion algorithms, and (j) is the result of our method. Please zoom in on the images to see more details. ![](./ComparedImages/LLVIP.png) 
## 3.4 M3FD
M3FD: Visual comparison results on the M3FD dataset. (a) and (b) are infrared and visible images, (c)-(i) are the results of image fusion algorithms, and (j) is the result of our method. Please zoom in on the images to see more details. ![](./ComparedImages/M3FD.png)
## 3.5 YOLOV7
YOLOV7: Visual comparison of object detection results with YOLOv7 algorithm. (a) and (b) are infrared and visible images, (c)-(i) are the detection results of image fusion algorithms, and (j) is the result of our method. Please zoom in on the images to see more details. ![](./ComparedImages/YOLOV7.png)

# 4. How to Run
## 4.1 Environment
      cuda	                    11.8	
      python	             3.10
      pytorch	             2.0
      imageio                   2.27.0
      kornia                    0.6.12
      numpy                     1.23.5
      opencv-python             4.7.0.72
      pillow                    9.5.0
      pytorch-msssim            0.2.1
      pyyaml                    6.0
      scikit-image              0.20.0
      scikit-learn              1.3.0
      scipy                     1.10.1
      tensorboard-data-server   0.7.0
      tensorboard-plugin-wit    1.8.1
      tqdm                      4.65.0 
      
Then pip the needed packages.

## 4.2 File Sturcture
      ---- Degradation Model (The proposed Degradation Model).
      ---- Model (The TeacherNet, StudentNet and Loss).
      ---- Test (The Test Files).
      ---- Train (The Train Files of TeacherNet and StudentNet).

## 4.3 How to Test
      1. cd to './Test'
      2. open './sys_configs_test.yaml' file to modify configuration
      3. python Test.py

## 4.4 Datasets
### 4.4.1 TestData
      The Test datasets are located in './Test/Dataets/...'.
### 4.4.2 TrainData
#### a. The Train dataset can be find in https://ivrlwww.epfl.ch/supplementary_material/cvpr11/index.html.
#### b. Remove misaligned images and corp them to 256x128 patches.
#### c. Naming Rule：
      ---- Train Data ROOT
            |--- 00001_IR.png
            |--- 00001_VI.png
              ....

## 4.4 How to Train
### 4.4.1 How to Train TeacherNet
      a. cd to ./Train.
      b. open './sys_configs_teacher.yaml' file to modify configuration.
      c. python Train_Teacher.py.

### 4.4.2 How to Train StudentNet
      a. cd to './Train'.
      b. open './sys_configs_student.yaml' file to modify configuration.
      c. python Train_Student.py.


# 5. Cite our work
If you like our work, you can cite us:

## 5.1 BibTex
@ARTICLE{10522962,
  author={Xue, Weimin and Liu, Yisha and Wang, Fei and He, Guojian and Zhuang, Yan},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={A Novel Teacher–Student Framework With Degradation Model for Infrared–Visible Image Fusion}, 
  year={2024},
  volume={73},
  number={},
  pages={1-12},
  keywords={Degradation;Image fusion;Training;Transformers;Task analysis;Image coding;Transform coding;Degradation model;infrared (IR)–visible (VIS) image fusion;IR–VIS object detection;teacher–student framework},
  doi={10.1109/TIM.2024.3398115}}
  
## 5.2 Plain Text
W. Xue, Y. Liu, F. Wang, G. He and Y. Zhuang, "A Novel Teacher–Student Framework With Degradation Model for Infrared–Visible Image Fusion," in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-12, 2024, Art no. 5020512, doi: 10.1109/TIM.2024.3398115.
keywords: {Degradation;Image fusion;Training;Transformers;Task analysis;Image coding;Transform coding;Degradation model;infrared (IR)–visible (VIS) image fusion;IR–VIS object detection;teacher–student framework},




