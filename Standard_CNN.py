import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
import time
from heatmap import *
from keras import backend as k
from livelossplot.keras import PlotLossesCallback
from PIL import Image
from keras.utils import plot_model

# File paths
TRAINING_LOGS_FILE = "Standard_Final_150_150.csv"
MODEL_SUMMARY_FILE = "Standard_Final_150_150.txt"
train_data_dir = 'C:/Users/YouWont4GetMe/Desktop/Dissertation/Programming/DissertationProject/data/Train_4.5K'
valid_data_dir = 'C:/Users/YouWont4GetMe/Desktop/Dissertation/Programming/DissertationProject/data/Validate'

# tensorboard graph plotter
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# hyperparameters
img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 281
validation_steps = 32
nb_filters1 = 16
nb_filters2 = 16
nb_filters3 = 32
nb_filters4 = 32
conv1_size = 3
conv2_size = 3
pool_size = 2
output_size = 2
lr = 0.001
adam = optimizers.adam(lr=lr)
Epochs = 5


# loss function
loss_function_multi = 'categorical_crossentropy'
# class mode
class_mode_multi = 'categorical'
# types of classes
classes = ['Nuclei', 'rbc']
classes_num = len(classes)


# CNN Model
def create_model():
    model = Sequential()

    # convolutional layer
    model.add(Conv2D(nb_filters1, (conv1_size, conv2_size), input_shape=(img_width, img_height, 3)))
    # activation layer relu
    model.add(Activation('relu'))
    # max pool layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # convolutional layer
    model.add(Conv2D(nb_filters2, (conv1_size, conv2_size)))
    # activation layer relu
    model.add(Activation('relu'))
    # max pool layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # convolutional layer
    model.add(Conv2D(nb_filters3, (conv1_size, conv2_size)))
    # activation layer
    model.add(Activation('relu'))
    # max pool layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # convolutional layer
    model.add(Conv2D(nb_filters4, (conv1_size, conv2_size)))
    # activation layer relu
    model.add(Activation('relu'))
    # max pool layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten
    model.add(Flatten())
    # compression
    model.add(Dense(256))
    # activation layer relu
    model.add(Activation('relu'))

    # compression
    model.add(Dense(output_size))
    # activation softmax
    model.add(Activation('softmax'))

    model.compile(loss=loss_function_multi, optimizer=adam, metrics=['accuracy'])

    print('model complied!!')

    with open(MODEL_SUMMARY_FILE, "w") as fh:
        model.summary(print_fn=lambda line: fh.write(line + "\n"))

    return model

# train the model
def train():
    # create model
    model = create_model()
    # load data
    data_generator = ImageDataGenerator(rescale=1./255)
    # train data
    train_generator = data_generator.flow_from_directory(directory=train_data_dir,
                                                         target_size=(img_width, img_height),
                                                         classes=classes,
                                                         class_mode=class_mode_multi,
                                                         batch_size=batch_size)

    # validate data
    validation_generator = data_generator.flow_from_directory(directory=valid_data_dir,
                                                              target_size=(img_width, img_height),
                                                              classes=classes,
                                                              class_mode=class_mode_multi,
                                                              batch_size=batch_size)
    print('starting training....')
    training = model.fit_generator(generator=train_generator,
                                   steps_per_epoch=samples_per_epoch,
                                   epochs=Epochs,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps,
                                   shuffle=True,
                                   callbacks=[tbCallBack, CSVLogger(TRAINING_LOGS_FILE, append=False, separator=";")])  # insert before cvs PlotLossesCallback(),

    # save weights
    model.save_weights('models/Standard_Final_150_150.h5')


# Load test image
def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


# Heatmap
def display_heatmap(weights_path, new_model, img_path, ids, width, height, preprocessing=None):
    # The quality is reduced.
    # If you have more than 8GB of RAM, you can try to increase it.
    img = image.load_img(img_path, target_size=(height * 8, width * 8))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.  # imshow expects values in the range [0, 1]
    print(k.image_data_format())
    if preprocessing is not None:
        img = preprocessing(img)

    out = new_model.predict(img)

    heatmap = out[0]  # Removing batch axis.

    if k.image_data_format() == 'channels_first':
        heatmap = heatmap[ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=0)
    else:
        heatmap = heatmap[:, :, ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=2)

    plt.imshow(heatmap, interpolation="none")
    plt.show()


# Load Model weights from file
def load_trained_model_single_image(weights_path):
    # create model
    model = create_model()
    # load weights
    model.load_weights(weights_path)
    # image path
    # Nuclei_Simple.png
    # Nuclei_Standard.png
    # Nuclei_Complex.png
    # RBC_Simple.png
    # RBC_Standard.png
    # RBC_Complex.png
    img_path = 'Nuclei_Simple.png'
    label = 0

    # load a single image
    new_image = load_image(img_path)
    im = Image.open(img_path)
    width, height = im.size

    # check prediction
    print("The class model Predicts: ", model.predict_classes(new_image))
    print("The Percentages for Predictions: ", model.predict(new_image))

    new_model = to_heatmap(model)
    # RBC = Purple for heat
    # GBM = Yellow for heat
    display_heatmap(weights_path, new_model, img_path, label, width, height)


# Load Model weights from file
def load_trained_model(weights_path):
    # create model
    model = create_model()
    # load weights
    model.load_weights(weights_path)  # image path

    # storage of results
    test_results = []
    test_images_passed = []
    test_images_failed = []

    # the testing class
    testing_class = 0
    i = 0
    FILE = "C:\\Users\\YouWont4GetMe\\Desktop\\Dissertation\\Images\\Images for editing\\GBM_PNG_Testing\\output\\"
    # loop through images
    for root, dirs, files in os.walk(FILE):
        for filename in files:

            img_path = FILE + filename  # Cat
            # load a single image
            new_image = load_image(img_path)
            # check prediction

            # saves the results
            prediction = model.predict_classes(new_image)
            test_results.append(prediction)

            # out puting the class found in image
            if prediction == testing_class:
                test_images_passed.append(img_path)
            else:
                test_images_failed.append(img_path)
            i = i + 1

    # print the number of failed classifation images
    for image in test_images_failed:
        print(image)

    print(test_results.count(0))
    print(test_results.count(1))


# train data
# train()

# loading all test images
# load_trained_model("models/Standard_Final_150_150.h5")

# loading singular
load_trained_model_single_image("models/Standard_Final_150_150.h5")

