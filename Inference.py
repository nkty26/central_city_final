import numpy as np 
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D 
import cv2, os


class Inference:
    def __init__(self):
        pass  
    
    def infer_realtime(self, model, image_path, current_image):
        image_path = os.path.join("".join(image_path.split("__")[0]), current_image)
        image_path = image_path + '.jpg'
        print(image_path)
        print(f"IMAGE {current_image} INFERENCE COMPLETED. FILE PATH: {image_path}")
        if image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
            PREDICTION = self.predict_image(model, image_path)
            for i, prob in enumerate(PREDICTION):
                predicted_class = PREDICTION.argmax(axis=1)[0]
                predicted_probability = np.max(PREDICTION[0]) 
                print(f"=====================================>  PREDICTION: [{predicted_class}]  |  PREDICTION PROBABILITY: [{"{:.2f}".format(predicted_probability*100)}%]\n")
        return (predicted_class, predicted_probability)
    
    def infer_realtime2(self, model, image_path, current_image):
        print(f"IMAGE {current_image} INFERENCE COMPLETED. FILE PATH: {image_path}")
        if image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
            PREDICTION = self.predict_image(model, image_path)
            for i, prob in enumerate(PREDICTION):
                predicted_class = PREDICTION.argmax(axis=1)[0]
                predicted_probability = np.max(PREDICTION[0]) 
                print(f"=====================================>  PREDICTION: [{predicted_class}]  |  PREDICTION PROBABILITY: [{"{:.2f}".format(predicted_probability*100)}%]\n")
        return (predicted_class, predicted_probability)

    def predict_image(self, model, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        reshaped_image = preprocessed_image.reshape(1, 28, 28, 1)
        predictions = model.predict(reshaped_image)
        return predictions
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (28, 28))
        normalized = resized_img.astype('float32') / 255.0
        reshaped = normalized.reshape(28, 28, 1)
        return reshaped
    


class DatasetSplitter:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def split_custom_dataset(self):
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.images, self.labels, test_size=0.1, random_state=42
        )
        return train_images, val_images, train_labels, val_labels
    
class ModelHandler:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, train_images, train_labels, val_images, val_labels, save_path):
        num_classes = len(np.unique(train_labels))
        if self.model.layers[-1].units != num_classes:
            self.model.pop()
            self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        history = self.model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
        self.model.save(save_path)
        return self.model

class DatasetLoader:
    def __init__(self):
        pass  
    
    def preprocess_image(self,image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (28, 28))
        normalized = resized_img.astype('float32') / 255.0
        reshaped = normalized.reshape(28, 28, 1)
        return reshaped
    
    def load_custom_images(self,src_path):
        images = []
        labels = []
        for subdirectory in os.listdir(src_path):
            folder_path = os.path.join(src_path, subdirectory)
            if os.path.isdir(folder_path):
                for image_path in os.listdir(folder_path):
                    if image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
                        image_path = os.path.join(folder_path, image_path)
                        img = self.preprocess_image(image_path)
                        if img is not None:
                            images.append(img)
                            labels.append(int(subdirectory))  # Use the subfolder name as the label
        images = np.array(images)
        labels = np.array(labels)
        return images, labels