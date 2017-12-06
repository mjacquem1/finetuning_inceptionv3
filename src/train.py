from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
# from keras.preprocessing.image import ImageDataGenerator
from MyImageDataGenerator import *
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint

from inception_v3 import InceptionV3
from inception_v3 import preprocess_input
# from inception_resnet_v2 import InceptionResNetV2, preprocess_input

import os
import sys
import glob
import matplotlib.pyplot as plt

module_path = os.path.dirname(os.path.abspath(__file__))

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
NB_EPOCHS_TL = 2
NB_EPOCHS_FT = 5
NB_EPOCHS_FT2 = 10
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE_FT = 172
NB_IV3_LAYERS_TO_FREEZE_FT2 = 100
# NB_RIV2_LAYERS_TO_FREEZE_FT = 700


# The functions adds a new fully connected layer at the top of base_model. The output of this FC layer has
# nb_classes outputs
def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def add_new_last_layer_2_labels(base_model, nb_classes1, nb_classes2):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions1 = Dense(nb_classes1, activation='softmax')(x)
    predictions2 = Dense(nb_classes2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=[predictions1, predictions2])
    return model


# We use this function to setup the transfer learning. The transfer learning means that we freezes the weights of
# the base_model, and we train only the last FC layer that we have added using add_new_last_layer function
def setup_to_transfer_learn(model, base_model):
    print("Nb layers: ", len(model.layers))
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=RMSprop(lr=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


# We use this function to setup the fine-tuning. The fine-tuning we freeze some layers at the end of our model,
# and we RETRAIN the rest of the layers
def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top
      layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in
         the inceptionv3 architecture
    Args:
     model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE_FT]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE_FT:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def setup_to_finetune2(model):
	"""Freeze the bottom NB_IV3_LAYERS and retrain the remaining top
	  layers.
	note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in
	     the inceptionv3 architecture
	Args:
	 model: keras model
	"""
	for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE_FT2]:
		layer.trainable = False
	for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE_FT2:]:
		layer.trainable = True
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def get_nb_jpgs(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*.jpg")))
    return cnt


def train(train_dir, val_dir, output_model_file, nb_epoch, batch_size, verbose=True):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_jpgs(train_dir)
#   nb_classes = len(glob.glob(train_dir + "/*"))
    nb_classes1 = 3
    nb_classes2 = 8
    nb_val_samples = get_nb_jpgs(val_dir)
    nb_epoch = NB_EPOCHS_TL if not nb_epoch else int(nb_epoch)
    batch_size = BAT_SIZE if not batch_size else batch_size

    # data preparation
    if verbose: print("data preparation...")
    # We prepare our data using data augmentation
    # Here, we apply multiple transformation to have a bigger dataset for training
    # for example we add zooms, flips, shifts
    train_datagen = MyImageDataGenerator(
#        preprocessing_function=preprocess_input,
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=45,
        width_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False
    )

    # we do the same transformation for the validation dataset
    valid_datagen = MyImageDataGenerator(
#        preprocessing_function=preprocess_input,
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=45,
        width_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False
    )

    # We generate now data from train_dir using the defined transformations
    train_generator = train_datagen.flow_from_directory_multilabel(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size
    )
    # We generate data from valid_dir using the defined transformations
    validation_generator = valid_datagen.flow_from_directory_multilabel(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size
    )

    # setup model
    if verbose: print("setup model...")
    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False => excludes final FC layer
#    base_model = InceptionResNetV2(weights='imagenet', include_top=False)  # include_top=False => excludes final FC layer
    model = add_new_last_layer_2_labels(base_model, nb_classes1, nb_classes2)

    # transfer learning
    if verbose: print("transfer learning...")
    setup_to_transfer_learn(model, base_model)

    '''
    ModelCheckPoint saves the model weights after each epoch if the validation loss decreased
    '''
    checkpointer = ModelCheckpoint(filepath=os.path.join(module_path, '../weights/weights_tl_tmp.h5'), verbose=0,
                                   save_best_only=True)

    # Train our model using transfer learning
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        epochs=NB_EPOCHS_TL,
        callbacks=[checkpointer],
        validation_data=validation_generator,
        validation_steps=nb_val_samples / batch_size,
        class_weight=['auto', 'auto']
    )

    # fine-tuning
    if verbose: print("fine-tuning...")
    setup_to_finetune(model)
    checkpointer = ModelCheckpoint(filepath=os.path.join(module_path, '../weights/weights_ft_tmp.h5'), verbose=0,
                                   save_best_only=True)
    # Train our model using fine-tuning
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        epochs=NB_EPOCHS_FT,
        callbacks=[checkpointer],
        validation_data=validation_generator,
        validation_steps=nb_val_samples / batch_size,
        class_weight=['auto', 'auto']
    )

    # fine-tuning 2 
    if verbose: print("fine-tuning 2...")
    setup_to_finetune2(model)
    checkpointer = ModelCheckpoint(filepath=os.path.join(module_path, '../weights/weights_ft2_tmp.h5'), verbose=0,
								   save_best_only=True)
    # Train our model using fine-tuning 2
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        epochs=NB_EPOCHS_FT2,
        callbacks=[checkpointer],
        validation_data=validation_generator,
        validation_steps=nb_val_samples / batch_size,
        class_weight=['auto', 'auto']
    )
    # Saving the model
    if verbose: print("Saving model...")
    model.save(output_model_file)


def training(path_to_dataset, output_model_file="weights/weights.h5", nb_epoch=NB_EPOCHS_TL, batch_size=BAT_SIZE):
    train_dir = path_to_dataset + "/train"
    val_dir = path_to_dataset + "/valid"

    if path_to_dataset is None:
        print("Please specify the path to your dataset!")
        print("Help: your dataset folder should contain two folders: 'train' and 'test'.")
        sys.exit(1)

    assert os.path.isdir(path_to_dataset)

    if not os.path.isdir(train_dir):
        print("train folder is not available!")
        sys.exit(1)

    if not os.path.isdir(val_dir):
        print("test folder is not available!")
        sys.exit(1)

    train(train_dir, val_dir, output_model_file, nb_epoch, batch_size)


if __name__ == "__main__":
#   training("data/cats-dogs")
#   training("../imdb_crop/data")
#   training("../openu/data2")
    training("../imdb_crop/data2")
