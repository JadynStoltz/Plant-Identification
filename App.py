import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil

class PlantIdentificationApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Load the trained model
        self.model = self.load_model()

        # Define class names (update this list based on your model)
        self.class_names = [
            "AloeVera", "Banana", "Bilimbi", "Cantaloupe", "Cassava",
            "Coconut", "Corn", "Cucumber", "Curcuma", "Eggplant",
            "Galangal", "Ginger", "Guava", "Kale", "Longbeans",
            "Mango", "Melon", "Orange", "Paddy", "Papaya",
            "Peper chili", "Pineapple", "Pomelo", "Shallot", "Soybeans",
            "Spinach", "Sweet Potatoes", "Tobacco", "Waterapple", "Watermelon"
        ]

        # Initialize video capture
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # Create a canvas to display the video
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Capture button
        self.btn_snapshot = ttk.Button(window, text="Capture", command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Result label
        self.result_label = ttk.Label(window, text="Prediction: ")
        self.result_label.pack(anchor=tk.CENTER, expand=True)

        # Update the video feed
        self.delay = 15
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close
        self.window.mainloop()

    def load_model(self):
        try:
            model = load_model('optimized_plant_classification_model.h5')  # Load the entire model
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            # Save the captured frame as an image
            image_path = "snapshot.jpg"
            cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.process_image(image_path)

    def process_image(self, image_path):
        if self.model is None:
            self.result_label.config(text="Error: Model not loaded.")
            return

        try:
            # Load and preprocess the image
            img = Image.open(image_path)
            img = img.resize((160, 160))  # Resize to match your model's input size
            img_array = np.array(img) / 255.0  # Normalize pixel values

            # Ensure the image has 3 channels (RGB)
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                img_array = np.stack([img_array] * 3, axis=-1)

            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = self.model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100

            # Show prediction result and ask for confirmation
            result = f"Prediction: {self.class_names[predicted_class]} ({confidence:.2f}% confident)"
            self.result_label.config(text=result)
            self.ask_for_confirmation(image_path, self.class_names[predicted_class])
        except Exception as e:
            self.result_label.config(text=f"Error during prediction: {e}")

    def ask_for_confirmation(self, image_path, predicted_class):
        response = simpledialog.askstring("Confirm Prediction",
                                          f"Is this plant {predicted_class}? (yes/no)")
        if response is not None:
            response = response.lower()
            if response == 'yes':
                self.add_to_database(image_path, predicted_class)
            elif response == 'no':
                self.choose_correct_class(image_path)
            else:
                messagebox.showerror("Error", "Invalid response. Please type 'yes' or 'no'.")

    def choose_correct_class(self, image_path):
        class_choice = simpledialog.askstring("Choose Correct Class",
                                              f"Select the correct class from the list:\n{', '.join(self.class_names)}")
        if class_choice is not None and class_choice in self.class_names:
            self.add_to_database(image_path, class_choice)
        else:
            messagebox.showerror("Error", "Invalid class name. Please select from the provided list.")

    def add_to_database(self, image_path, class_name):
        # Create a directory for the class if it doesn't exist
        if not os.path.exists(class_name):
            os.makedirs(class_name)

        # Move or copy the image to the class directory
        shutil.copy(image_path, os.path.join(class_name, os.path.basename(image_path)))

        messagebox.showinfo("Success", f"Image added to the {class_name} class successfully!")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Convert the frame to a format suitable for Tkinter
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def on_closing(self):
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    PlantIdentificationApp(tk.Tk(), "Plant Identification App")
