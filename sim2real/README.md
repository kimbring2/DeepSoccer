# Code description
Simulation to Real part consists of two step. The first step is to segment an important part from the real image. 

ADE20K dataset is used to segment image using [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/). 
Download dataset and move it under your workspace. You need to set that path in [segmentation.ipynb](https://github.com/kimbring2/DeepSoccer/blob/master/sim2real/segmentation.ipynb) file.

You are ready to train Segmentation model. That model will be used to separate floor from image. However, classic OpenCV method 
is used to get ball and goalpost part because they have well-marked color such as green, red.

You can check how to run it together in file. Try to segment [content_image](https://drive.google.com/drive/folders/1TuaYWI191L0lc4EaDm23olSsToEQRHYY?usp=sharing) to. It will be used at next step.

Next step is to convert real image to simulation image using CycleGAN. The dataset for training CycleGAN should become set of image which include goal, goalpost, floor, and wall from various view angle. The reason we applied segmention before this is to remove unrelated object. Please place [segmented content_image](https://drive.google.com/drive/folders/1S4R7NGOu-IZZskSwGL5YXpU7-fVQLSqR?usp=sharing), [style image](https://drive.google.com/drive/folders/166qiiv2Wx0d6-DZBwHiI7Xgg6r_9gmfy?usp=sharing) to your workspace.

You need to set that path in [cyclegan.ipynb](https://github.com/kimbring2/DeepSoccer/blob/master/sim2real/cyclegan.ipynb) file. You are ready to train CycleGAN model. Run each cell of Notebook. 

Trained model will be copied to Jetson board to convery camera image to image of simulation for Sim2Real.

# Software Dependency
- Tensorflow 2.2.0
- opencv-python
