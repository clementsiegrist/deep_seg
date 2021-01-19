import tensorflow.keras as keras
from unet_pipeline.generator import load_pretrained, train_val_gen
from unet_pipeline.generator import predict_on_test_and_plot
from unet_pipeline.generator import plot_results

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
nb_epochs = 10
model_name = 'cell_seg.hdf5'


def main(train_path, train_batch_size, validation_path, validation_batch_size,
         nb_epochs):  # , test_path): #model_name):

    checkpoint_path = '/content/weights'
    test_path = "/content/drive/MyDrive/Colab Notebooks/cells/test"
    model_name = 'cell_seg.hdf5'
    train_generator, validation_generator, train_image_generator, validation_image_generator = train_val_gen(train_path,
                                                                                                             train_batch_size,
                                                                                                             validation_path,
                                                                                                             validation_batch_size)
    model = load_pretrained(checkpoint_path, train_generator, validation_generator, validation_batch_size,
                            train_batch_size, nb_epochs, train_image_generator, validation_image_generator)
    # model = load_unet(train_path, train_generator, validation_generator, validation_batch_size, train_batch_size, nb_epochs, train_image_generator, validation_image_generator)
    # model = Unet_backbone(train_generator, train_image_generator, validation_generator, validation_image_generator, train_batch_size, validation_batch_size, nb_epochs)
    # model = load_weight_map_unet(train_path, train_generator, validation_generator, validation_batch_size, train_batch_size, nb_epochs, train_image_generator, validation_image_generator)
    plot_results(model)
    predict_on_test_and_plot(model_name=model_name, test_path=test_path, num=96)
    return model