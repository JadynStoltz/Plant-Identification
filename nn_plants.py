import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, \
    GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up paths
base_dir = "C:/Users/jadyn/Documents/Plant Identification/archive/split_ttv_dataset_type_of_plants"
train_dir = os.path.join(base_dir, 'Train_Set_Folder')
validation_dir = os.path.join(base_dir, 'Validation_Set_Folder')
test_dir = os.path.join(base_dir, 'Test_Set_Folder')

# Reduce image size
img_height, img_width = 160, 160
batch_size = 32

# Get class names from the training directory structure
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)


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

# Use MobileNetV2 as base model
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
initial_learning_rate = 0.001
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

# Get predictions
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_dataset.classes

# Calculate overall metrics
overall_accuracy = accuracy_score(y_true, y_pred_classes)
overall_precision = precision_score(y_true, y_pred_classes, average='weighted')
overall_recall = recall_score(y_true, y_pred_classes, average='weighted')
overall_f1 = f1_score(y_true, y_pred_classes, average='weighted')

print("\nOverall Performance Metrics:")
print(f"Accuracy: {overall_accuracy * 100:.2f}%")
print(f"Precision: {overall_precision * 100:.2f}%")
print(f"Recall: {overall_recall * 100:.2f}%")
print(f"F1-Score: {overall_f1 * 100:.2f}%")

# Print classification report
print("\nDetailed Classification Report:")
class_report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
for class_name, metrics in class_report.items():
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision'] * 100:.2f}%")
        print(f"  Recall: {metrics['recall'] * 100:.2f}%")
        print(f"  F1-Score: {metrics['f1-score'] * 100:.2f}%")
        print(f"  Support: {metrics['support']}")

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
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

# Analyze model performance
class_accuracies = cm.diagonal() / cm.sum(axis=1)
best_classes = np.argsort(class_accuracies)[-5:][::-1]
worst_classes = np.argsort(class_accuracies)[:5]

print("\nTop 5 best-performing classes:")
for i in best_classes:
    print(f"{class_names[i]}: {class_accuracies[i] * 100:.2f}%")

print("\nTop 5 worst-performing classes:")
for i in worst_classes:
    print(f"{class_names[i]}: {class_accuracies[i] * 100:.2f}%")

# Calculate and print additional statistics
print("\nAdditional Statistics:")
print(f"Number of classes: {num_classes}")
print(f"Total number of test samples: {len(y_true)}")
print(f"Average samples per class: {len(y_true) / num_classes:.2f}")
print(f"Class balance (min/max ratio): {min(cm.sum(axis=1)) / max(cm.sum(axis=1)):.2f}")

# Calculate and print per-class error rates
error_rates = 1 - class_accuracies
print("\nPer-class error rates:")
for i, error_rate in enumerate(error_rates):
    print(f"{class_names[i]}: {error_rate * 100:.2f}%")

# Print model summary
print("\nModel Architecture:")
model.summary()

# Save the model
model.save('optimized_plant_classification_model.h5')

print("\nModel analysis complete. Check 'confusion_matrix.png' and 'training_history.png' for visualizations.")