import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set image parameters
img_height, img_width = 224, 224  # Target size for all images

# Load the saved model
model = tf.keras.models.load_model('optimized_plant_classification_model.h5')

# Get class names from the test directory
test_dir = "C:/Users/jadyn/Documents/Plant Identification/archive/split_ttv_dataset_type_of_plants/Test_Set_Folder"
class_names = os.listdir(test_dir)


# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to evaluate the model on the test set
def evaluate_model(test_dir):
    all_predictions = []
    all_true_labels = []

    # Traverse through each class folder in the test directory
    for class_name in class_names:
        class_folder = os.path.join(test_dir, class_name)

        # Check if the path is a directory
        if os.path.isdir(class_folder):
            # Iterate through each image in the class folder
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)

                # Load and preprocess the image
                img_array = load_and_preprocess_image(img_path)

                # Make prediction
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction[0])]

                # Append the predicted and true labels
                all_predictions.append(predicted_class)
                all_true_labels.append(class_name)

    # Calculate accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f'Accuracy of the model on the test set: {accuracy:.2f}')


# Evaluate the model
evaluate_model(test_dir)