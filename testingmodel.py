from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


def create_simple_model():
    model = Sequential([
        Flatten(input_shape=(160, 160, 3)),
        Dense(128, activation='relu'),
        Dense(30, activation='softmax')
    ])

    model.summary()
    return model


# Create and save a simple model
simple_model = create_simple_model()
simple_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save the simple model
try:
    simple_model.save('simple_model.h5')
    print("Simple model saved successfully!")
except Exception as e:
    print(f"Error saving simple model: {e}")
