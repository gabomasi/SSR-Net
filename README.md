# SSR-Net
**[IJCAI18] SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation**
+ A real-time age estimation model with 0.32MB.
+ Gender regression is also added!
+ Megaage-Asian is provided in https://github.com/b02901145/SSR-Net_megaage-asian
+ Coreml model (0.17MB) is provided in https://github.com/shamangary/Keras-to-coreml-multiple-inputs-example

**Code Author: Tsun-Yi Yang**

**Last update: 2018/08/06 (Adding megaage_asian training. Typo fixed.)**
### Real-time webcam demo

<img src="https://media.giphy.com/media/AhXTjtGO9tnsyi2k6Q/giphy.gif" height="240"/> <img src="https://github.com/shamangary/SSR-Net/blob/master/age_gender_demo.png" height="240"/>

## Paper

### PDF
https://github.com/shamangary/SSR-Net/blob/master/ijcai18_ssrnet_pdfa_2b.pdf

## Dependencies
```
pip install -r requirements.txt
```
## Video demo section
Pure CPU demo command:
```
cd ./demo
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_mtcnn.py VIDEO.mp4

# Or you can use

KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_mtcnn.py VIDEO.mp4 '3'
```
+ Note: You may choose different pre-trained models. However, the morph2 dataset is under a well controlled environment and it is much more smaller than IMDB and WIKI, the pre-trained models from morph2 may perform ly on the in-the-wild images. Therefore, IMDB or WIKI pre-trained models are recommended for in-the-wild images or video demo.

+ We use dlib detection and face alignment in the previous experimental section since the face data is well organized. However, dlib cannot provide satisfactory face detection for in-the-wild video. Therefore we use mtcnn as the detection process in the demo section.

## Real-time webcam demo

Considering the face detection process (MTCNN or Dlib) is not fast enough for real-time demo. We show a real-time webcam version by using lbp face detector.

```
cd ./demo
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_ssrnet_lbp_webcam.py
```
+ Note that the covered region of face detection is different when you use MTCNN, Dlib, or LBP. You should choose similar size between the inference and the training.

```
Note that the score can be between [0,1] and the 'V' inside SSR-Net can be changed into 1 for general propose regression.
```


## Third Party Implementation
MXNET:
https://github.com/wayen820/gender_age_estimation_mxnet
