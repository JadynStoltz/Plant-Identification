import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set up paths
base_dir = r"C:\Users\jadyn\Documents\Plant Identification\PlantNet300k\plantnet_300K\images"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
json_path = r"C:\Users\jadyn\Documents\Plant Identification\PlantNet300k\plantnet_300K\plantnet300k_species_id_2_name.json"

# Load class names from JSON file
with open(json_path, 'r') as f:
    class_dict = json.load(f)

class_names = list(class_dict.values())
num_classes = len(class_names)

# Image parameters
img_height, img_width = 160, 160
batch_size = 32

def create_dataset(directory, is_training=False):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 if is_training else 0
    )

    dataset = datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=is_training,
        subset='training' if is_training else None
    )

    return dataset

# Create datasets
train_dataset = create_dataset(train_dir, is_training=True)
validation_dataset = create_dataset(validation_dir)
test_dataset = create_dataset(test_dir)

# Load the pre-trained model
base_model = load_model('optimized_plant_classification_model.h5')

# Check if the model is sequential
if isinstance(base_model, tf.keras.Sequential):
    # Create a new model using the layers from the pre-trained model
    inputs = Input(shape=(img_height, img_width, 3))
    x = inputs
    for layer in base_model.layers:
        x = layer(x)

    # Add a new dense layer for the 1081 classes with a unique name
    outputs = Dense(num_classes, activation='softmax', name="new_dense")(x)

    # Create the new model
    model = Model(inputs=inputs, outputs=outputs)

else:
    # If the model is not Sequential, it should have input and output defined already
    inputs = base_model.input
    x = base_model.output

    # Add a new dense layer for the 1081 classes with a unique name
    outputs = Dense(num_classes, activation='softmax', name="new_dense")(x)

    # Create the new model
    model = Model(inputs=inputs, outputs=outputs)

# Freeze all layers except the last one
for layer in model.layers[:-1]:
    layer.trainable = False

# Compile the model
initial_learning_rate = 0.0001
optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[reduce_lr, early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Save the updated model
model.save('plantnet300k_classification_model.h5')

print("Model training complete. New model saved as 'plantnet300k_classification_model.h5'.")

# Generate predictions
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_dataset.classes

# Generate classification report
report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Create a DataFrame for species statistics
species_stats = pd.DataFrame({
    'Species': class_names,
    'Accuracy': class_accuracy,
    'Precision': [report[name]['precision'] for name in class_names],
    'Recall': [report[name]['recall'] for name in class_names],
    'F1-Score': [report[name]['f1-score'] for name in class_names],
    'Support': [report[name]['support'] for name in class_names]
})

# Calculate overall statistics
overall_stats = pd.DataFrame({
    'Metric': ['Accuracy', 'Macro Avg Precision', 'Macro Avg Recall', 'Macro Avg F1-Score'],
    'Value': [
        report['accuracy'],
        report['macro avg']['precision'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score']
    ]
})

# Calculate training statistics
train_stats = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Train Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Train Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})

# Create an Excel writer object
with pd.ExcelWriter('plantnet300k_model_stats.xlsx') as writer:
    # Write each DataFrame to a different sheet
    species_stats.to_excel(writer, sheet_name='Species Statistics', index=False)
    overall_stats.to_excel(writer, sheet_name='Overall Statistics', index=False)
    train_stats.to_excel(writer, sheet_name='Training Statistics', index=False)

    # Write confusion matrix
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    cm_df.to_excel(writer, sheet_name='Confusion Matrix')

# Plot confusion matrix
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

print("Statistics saved to 'plantnet300k_model_stats.xlsx'")
print("Confusion matrix visualization saved as 'confusion_matrix.png'")
print("Training history plot saved as 'training_history.png'")
