import numpy as np
import os
import cv2
from keras.preprocessing import image

from inception_v3 import InceptionV3
from inception_v3 import preprocess_input
from train import add_new_last_layer


def testing(weights_path="weights/weights.h5"):
	classes = ["cat", "dog"]
	classes = np.sort(classes)
	nb_classes = len(classes)
	# Setup the inceptionV3 model, pretrained on ImageNet dataset, without the fully connected part.
	base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
	# Add a new fully connected layer at the top of the base model. The weights of this FC layer are random
	# so they need to be trained
	model = add_new_last_layer(base_model, nb_classes)
	# We have already trained our model, so we just need to load it
	model.load_weights(weights_path)
	# Here, instead of writing the path and load the model each time, we load our model one time and we make a loop
	# where we ask only for the image path every time. If we enter "stop", we exit the loop
	img_path = input("new image? ")
	while img_path != "stop":
		if os.path.isfile(img_path):
			img = image.load_img(img_path, target_size=(299, 299))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			preds = model.predict(x)
			# decode the results into a list of tuples (class, description, probability)
			# (one such list for each sample in the batch)
			label = classes[np.argmax(preds)]
			p = preds[0][np.argmax(preds)] * 100
			print("Label: {}, {:.2f}%".format(label, p))
			# orig = cv2.imread(img_path)
			# cv2.putText(orig, "Label: {}, {:.2f}%".format(label, p),
			# 			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
			# cv2.imshow("Classification", orig)
			# cv2.waitKey(0)

		else:
			print("file doesn't exist!")
		img_path = input("new image? ")


if __name__ == "__main__":
	testing(weights_path="weights/inceptionv3_catsdogs_weights_10epochs.h5")
