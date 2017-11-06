# finetuning_inceptionv3

This an example of finetuning the inceptionV3 model.

Paper: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826).

# Data
Your data must have the same form as the one here:
dataset/train
-------------/class1
--------------------/image_1_1.jpg
--------------------/image_1_2.jpg
...
-------------/class2
--------------------/image_2_1.jpg
--------------------/image_2_2.jpg
...
...
...
-------------/classN
--------------------/image_N_1.jpg
--------------------/image_N_2.jpg
...

# Train
To train your model, you can change some parameters like:
NB_EPOCHS = 5
BAT_SIZE = 32

To launch the training, go to the  project folder (fintuning_inceptionv3) and do:
python src/train.py

# Test
To test your model on some images, go to src/test.py, set the path to your model, save and from the terminal do:
python src/test.py


(o) A.I. Mons team, www.ai-mons.com

