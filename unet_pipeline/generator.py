from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from keras.optimizers import Adam
import keras
from deep_seg.unet_pipeline.unet import CorneaNet

train_path = "/content/drive/MyDrive/Colab Notebooks/cells/train"
validation_path = "/content/drive/MyDrive/Colab Notebooks/cells/validation"
test_path = "/content/drive/MyDrive/Colab Notebooks/cells/test"
test_batch_size = 4
checkpoint_path = '/content/weights'
input_size = (256, 256)
target_size = (256, 256) # (height, width) of input_size should be divisible by 32
num_class = 2
# fine tuning
train_batch_size = 4
test_batch_size = 4
validation_batch_size = 4
learning_rate = 1e-4
nb_epochs = 5

def adjust_data(img, mask, num_class):
    # modify the masks to be one-hot encoded
    mask = mask[:, :, :, 0] if(len(mask.shape) == 4) else mask[:, :, 0]
    new_mask = np.zeros(mask.shape + (num_class,), dtype=np.uint8)
    for i in range(num_class):
        new_mask[mask == i, i] = 1
    mask = new_mask
    return img, mask

def generator_adjusted(generator, num_class):
    for (img, mask) in generator:
        img, mask = adjust_data(img, mask, num_class)
        yield img, mask


def plot_results(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def train_val_gen(train_path, train_batch_size, validation_path, validation_batch_size):
    # generator for training images
    train_image_data_gen_args = dict(rescale=1. / 255)  # transformations
    train_image_datagen = ImageDataGenerator(**train_image_data_gen_args)
    train_image_generator = train_image_datagen.flow_from_directory(train_path, classes=['images'], class_mode=None,
                                                                    # no labels are returned
                                                                    color_mode='rgb', target_size=input_size,
                                                                    batch_size=train_batch_size, save_to_dir=None,
                                                                    save_prefix='image',
                                                                    seed=1)

    # generator for training masks
    train_mask_data_gen_args = dict()  # transformations
    train_mask_datagen = ImageDataGenerator(**train_mask_data_gen_args)
    train_mask_generator = train_mask_datagen.flow_from_directory(train_path,
                                                                  classes=['masks'], class_mode=None,
                                                                  # no labels are returned
                                                                  color_mode='grayscale', target_size=input_size,
                                                                  batch_size=train_batch_size, save_to_dir=None,
                                                                  save_prefix='mask',
                                                                  seed=1)

    # generator for validation images
    validation_image_data_gen_args = dict(rescale=1. / 255)  # transformations
    validation_image_datagen = ImageDataGenerator(**validation_image_data_gen_args)
    validation_image_generator = validation_image_datagen.flow_from_directory(validation_path, classes=['images'],
                                                                              class_mode=None,  # no labels are returned
                                                                              color_mode='rgb', target_size=input_size,
                                                                              batch_size=validation_batch_size,
                                                                              save_to_dir=None, save_prefix='image',
                                                                              seed=1)

    # generator for validation masks
    validation_mask_data_gen_args = dict()  # transformations
    validation_mask_datagen = ImageDataGenerator(**validation_mask_data_gen_args)
    validation_mask_generator = validation_mask_datagen.flow_from_directory(validation_path,
                                                                            classes=['masks'], class_mode=None,
                                                                            # no labels are returned
                                                                            color_mode='grayscale',
                                                                            target_size=input_size,
                                                                            batch_size=validation_batch_size,
                                                                            save_to_dir=None, save_prefix='mask',
                                                                            seed=1)

    # generator for training (image, mask)
    train_generator = zip(train_image_generator, train_mask_generator)
    train_generator = generator_adjusted(train_generator, num_class)  # modify the masks to be one-hot encoded

    # generator for training (image, mask)
    validation_generator = zip(validation_image_generator, validation_mask_generator)
    validation_generator = generator_adjusted(validation_generator, num_class)  # modify the masks to be one-hot encoded

    return train_generator, validation_generator, train_image_generator, validation_image_generator


# generator for test images
def predict_on_test_and_plot(test_path, model_name, num):
    test_image_data_gen_args = dict(rescale=1. / 255)  # transformations
    test_image_datagen = ImageDataGenerator(**test_image_data_gen_args)
    test_image_generator = test_image_datagen.flow_from_directory(test_path, classes=['images'], class_mode=None,
                                                                  # no labels are returned
                                                                  color_mode='rgb', target_size=input_size,
                                                                  batch_size=1, shuffle=False, seed=1)
    # generator for test masks
    test_mask_data_gen_args = dict()  # transformations
    test_mask_datagen = ImageDataGenerator(**test_mask_data_gen_args)
    test_mask_generator = test_mask_datagen.flow_from_directory(test_path,
                                                                classes=['masks'], class_mode=None,
                                                                # no labels are returned
                                                                color_mode='grayscale', target_size=input_size,
                                                                batch_size=1, shuffle=False, seed=1)

    # model = load_model(os.path.join(train_path, 'checkpoint', model_name), custom_objects={'lossFunc': WeightedLoss})
    model = load_model(os.path.join('/content/weights', 'checkpoint', model_name), compile=False)
    # model.compile(optimizer=Adam(lr=learning_rate),
    #              loss=weightedLoss(keras.losses.categorical_crossentropy, class_weight),
    #              metrics=['accuracy'])

    num_valid = test_image_generator.samples - num
    results = model.predict_generator(test_image_generator, num_valid, verbose=1)
    results = results.astype(float)
    results = np.max(results, axis=3)
    # print(results[])

    '''
    model = unet(num_class, input_size=input_size)
    model.load_weights(os.path.join(train_path, 'checkpoint', 'unet_segmentation_epiderm.hdf5'))
    num_valid = test_image_generator.samples
    results = model.predict_generator(test_image_generator, num_valid, verbose=1)
    results = np.argmax(results, axis=3)
    '''
    # Display result
    fig1, ax1 = plt.subplots(num_valid, figsize=(4, 4), constrained_layout=True)
    fig2, ax2 = plt.subplots(num_valid, figsize=(4, 4), constrained_layout=True)
    fig3, ax3 = plt.subplots(num_valid, figsize=(4, 4), constrained_layout=True)
    # for i in range(5):
    # img, mask = next(zip(test_image_generator, test_mask_generator))
    for i, (img, mask) in enumerate(zip(test_image_generator, test_mask_generator)):
        if i >= num_valid:
            break
        ax1[i].imshow(img[0, :, :, :])  # original
        ax1[i].axis('off')
        ax2[i].imshow(img[0, :, :, :])  # original
        ax2[i].contour(mask[0, :, :, 0], colors='c', linewidths=0.5)
        ax2[i].contour(results[i].astype('float'), colors='r', linewidths=0.3)
        ax2[i].axis('off')
        ax3[i].imshow(results[i], 'gray')  # original
        ax3[i].axis('off')


def load_pretrained(checkpoint_path, train_generator, validation_generator, validation_batch_size, train_batch_size,
                    nb_epochs,
                    train_image_generator, validation_image_generator):
    model = CorneaNet(pretrained_weights=None, input_size=(256, 256, 3))
    # pretrained_weights = os.path.join(train_path, 'checkpoint', 'unet_segmentation_epiderm.hdf5')
    # model.load_weights(pretrained_weights)
    model.summary()
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_path, 'checkpoint', 'cell_seg.hdf5'),
        monitor='loss',
        verbose=1,
        save_best_only=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, verbose=1, mode='max')

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_image_generator.samples // train_batch_size,
                                  epochs=nb_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_image_generator.samples // validation_batch_size,
                                  callbacks=[model_checkpoint, rlr])
    return history