# import all necessary packages for this script
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Argument parser that will allow the python script to be launched from terminal
# --dataset  shows the direction to input dataset of faces with and without mask
# --plot  shows direction to output training history plot
# --model  shows direction to resulting serialized face mask model
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# Set the number of initial learning rate, number of epochs that will be trained for and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


# DATA PRE-PROCESSING
# Begin to pre-process the training data by grabbing the list of images and later initializing the list
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))   # Grab the images in dataset
data = []   # Initialize data list
labels = []   # Initialize label list

# Loop over the imagepaths and simultaneously pre-process the images
for imagePath in imagePaths:
	# Extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# Load the input image
	image = load_img(imagePath, target_size=(224, 224))   # Resize image to 224x224 pixels
	image = img_to_array(image)   # Convert images to array format
	image = preprocess_input(image)   # Scale pixel intensities to desired range [-1,1]

	# Append the pre-processed image to data list and the associated label to labels list
	data.append(image)
	labels.append(label)

# Ensure the training data and labels are in NumPy array format
data = np.array(data, dtype="float32")
labels = np.array(labels)


# PREPARATION FOR DATA AUGMENTATION
# Perform one-hot encoding on the labels, each element in labels array consists of an array in which only one index is 'hot'
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Create a partition to split data into 80% train data and 20% test data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


# PREPARE MOBILENETV2 FOR FINE-TUNING
# Load MobileNet with pre-trained ImageNet weights, leaving off head FC of the network
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Construct a new FC head that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Append the newly constructed FC head to the base in place of the old head
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the base layers of the network by looping over all layers in the base model
# This means the weight of base layers will not be updated during first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compile the model with the Adam optimizer
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Find the index of the label with corresponding largest predicted probability for each image in the testing set
predIdxs = np.argmax(predIdxs, axis=1)

# Show classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# Plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


# To use this script input the code below to a terminal
#python mask_train.py --dataset images_dataset