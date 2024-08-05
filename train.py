import imgaug.augmenters as iaa
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.api.models import load_model
from keras.api.utils import to_categorical
from keras.api.optimizers import SGD, Adam 
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Activation, ReLU, LeakyReLU, Softmax, Dropout, Conv2D, MaxPooling2D 
import cv2, os, json
from sklearn.utils import class_weight 
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve 
import seaborn as sns 

class DatasetLoader:
    def __init__(self):
        pass  

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
                            # print("LABEL      ", int(subdirectory), "   |   IMAGE     ", image_path)
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
      
    def preprocess_image(self,image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (28, 28))
        normalized = resized_img.astype('float32') / 255.0
        reshaped = normalized.reshape(28, 28, 1)
        # kernel = np.ones((3,3), np.uint8)  
        # eroded = cv2.erode(reshaped, kernel, iterations=1)
        # opened = cv2.dilate(eroded, kernel, iterations=1)
        # DatasetLoader.render(opened, "opened")
        return reshaped
    
    def render(image,string):
        cv2.imshow(string, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class DatasetProcessor:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def split_custom_dataset(self):
        # train_images, val_images, train_labels, val_labels = train_test_split(
        #     self.images, self.labels, test_size=0.1, random_state=42
        # )
        train_images, temp_images, train_labels, temp_labels = train_test_split(
        self.images, self.labels, test_size=0.2, random_state=42
        )

        val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42
        )
        return train_images, val_images, train_labels, val_labels, test_images, test_labels
    
    def handle_class_imbalance(self, src_path, dest_path):
        print("Handling class imbalance...")
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.labels), self.labels)
        class_weights = dict(enumerate(class_weights))
        with open(os.path.join(dest_path, 'class_weights.json'), 'w') as f:
            json.dump(class_weights, f)
        return class_weights
    
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
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, train_images, train_labels, val_images, val_labels, save_path, normalized_weights):
        num_classes = len(np.unique(train_labels))
        if self.model.layers[-1].units != num_classes:
            self.model.pop()
            self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        history = self.model.fit(train_images, train_labels, epochs=20, batch_size=32, class_weight = normalized_weights, validation_data=(val_images, val_labels))
        self.model.save(save_path)
        return history 
    
class Evaluator:    
    def __init__(self, model, test_images, test_labels):
        self.model = model 
        self.test_images = test_images
        self.test_labels = test_labels

    def get_predictions(self):
        y_scores = self.model.predict(self.test_images)
        y_pred = np.argmax(y_scores, axis=1)
        return y_pred, y_scores
    
    def plot_confusion_matrix(self, y_test_pred, y_test_true):
        num_classes = 10 
        CM = confusion_matrix(y_test_true, y_test_pred, labels=range(num_classes))
        plt.figure(figsize=(10, 10))
        sns.heatmap(CM, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        return CM
    
    def plot_classification_report(self, y_test_pred, test_labels):
        report = classification_report(test_labels, y_test_pred, output_dict=True)
        print(report)
        df_report = pd.DataFrame(report).transpose()
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Classification Report')
        plt.show()
        return report

    def plot_learning_curve(self,history):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.show()

    def plot_f1_score(self, y_true, y_scores, num_classes):
        f1_scores = []
        thresholds_list = []
        for i in range(num_classes):
            precision, recall, thresholds = precision_recall_curve(y_true == i, y_scores)
            f1_scores_class = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1_scores_class)
            thresholds_list.append(thresholds)
        plt.figure(figsize=(12, 8))
        for i in range(num_classes):
            plt.plot(thresholds_list[i], f1_scores[i], label=f'Class {i}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Threshold for Multi-Class Classification')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_inference_performance_test_dataset(self, model, test_images, test_labels):
        correct = 0 
        total = 0
        print("Testing Inference Performance Using Model : ", model)
        for image, ground_truth in zip(test_images, test_labels):
            total += 1
            preprocessed_image = image.reshape(1, 28, 28, 1)
            PREDICTION = model.predict(preprocessed_image)
            predicted_class = PREDICTION.argmax(axis=1)[0]
            predicted_probability = np.max(PREDICTION[0])
            print(predicted_class, ground_truth, predicted_probability)
            print(f"=====================================>  PREDICTION: [{predicted_class}]  |  GROUND_TRUTH: [{ground_truth}]  |  PREDICTION PROBABILITY: [{"{:.2f}".format(predicted_probability*100)}%]\n")
            if predicted_class == ground_truth:
                correct += 1 
        print(f"Inference Accuracy: {correct/total*100:.2f}%")
        
    def augment_images(self, test_images, test_labels):
        augmentations = [
            iaa.Sequential([iaa.GammaContrast((0.5, 2.0))]),                  # Rotation 0
            iaa.Sequential([iaa.Affine(rotate=(-30, 30))]),   
            iaa.Sequential([iaa.Affine(scale=(0.8, 1.2))]),                   # Scaling
            iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.05))]),      # Elastic distortions
            iaa.Sequential([iaa.GaussianBlur(sigma=(0, 3.0))]),  # Gaussian blur
            iaa.Sequential([iaa.GaussianBlur(sigma=(0, 2.0))]),  # Gaussian blur
            iaa.Sequential([iaa.GaussianBlur(sigma=(0, 1.0))]),  # Gaussian blur
            iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(0, 0.125))]),  # Gaussian noise
            iaa.Sequential([iaa.LinearContrast((0.5, 2.0))]),  # Contrast normalization
            iaa.Sequential([iaa.LinearContrast((1.0, 1.0))]),  # Contrast normalization
            iaa.Sequential([iaa.Affine(scale=1.2)]),  # Scaling
            iaa.Sequential([iaa.Affine(shear=16)]),  # Shear
            iaa.Sequential([iaa.GaussianBlur(sigma=0.5)]),  # Blur
            iaa.Sequential([iaa.LinearContrast(1.5)]),  # Contrast
            iaa.Sequential([iaa.Multiply(1.5)]),  # Brightness
            iaa.Sequential([iaa.ElasticTransformation(alpha=1.0, sigma=1.0)]),  # Elastic
            iaa.Sequential([iaa.PiecewiseAffine(scale=0.02)]),  # Piecewise Affine
            iaa.Sequential([iaa.Crop(percent=(0, 0.1))]),  # Random Croppingqqq
        ]

        augmentation_descriptions = [
            f"GammaContrast (gamma range = {0.5} to {2.0})",
            f"Affine (rotate range = {-30} to {30} degrees)",
            f"Affine (scale range = {0.8} to {1.2})",
            f"PiecewiseAffine (scale range = {0.01} to {0.05})",
            f"GaussianBlur (sigma range = {0} to {3.0})",
            f"GaussianBlur (sigma range = {0} to {2.0})",
            f"GaussianBlur (sigma range = {0} to {1.0})",
            f"AdditiveGaussianNoise (scale range = {0} to {0.125})",
            f"LinearContrast (alpha range = {0.5} to {2.0})",
            f"LinearContrast (alpha = 1.0, no change)",
            f"Affine (scale = 1.2)",
            f"Affine (shear = 16 degrees)",
            f"GaussianBlur (sigma = 0.5)",
            f"LinearContrast (alpha = 1.5)",
            f"Multiply (factor = 1.5)",
            f"ElasticTransformation (alpha = 1.0, sigma = 1.0)",
            f"PiecewiseAffine (scale = 0.02)",
            f"Crop (percent range = {0} to {0.1})"
        ]

        augmented_images_list = []
        for i in range(len(test_images)):
            image = test_images[i]
            if image.ndim == 3:  # (28, 28, 1)
                image = np.expand_dims(image, axis=0)  # (1, 28, 28, 1)
            for j, aug in enumerate(augmentations):
                augmented_image = aug(image=image[0])  # image[0] gives us (28, 28, 1)
                # self.render(augmented_image, f"augmented {i} {j}")
                augmented_images_list.append((np.array(augmented_image), test_labels[i]))
        return (augmented_images_list, augmentation_descriptions)

    def evaluate_inference_performance_augmented(self, model, augmentation_tuple, test_labels):
        correct = 0 
        total = 0
        aug_images = augmentation_tuple[0]
        aug_descriptions = augmentation_tuple[1]
        print("=====================================================================================================")
        print("                      Testing Augmented Inference Performance Using Model : ", model)
        print("=====================================================================================================")
        j = 0 
        num_augs = int(len(aug_images) // len(test_labels))
        correct = 0 
        total = 0
        avg_predicted_probability = 0
        incorrect = {}
        idx = 0 
        for i in range(len(aug_images)):
            idx+=1 
            if idx == num_augs:
                idx = 0
            total += 1 
            image = aug_images[i][0]
            ground_truth = aug_images[i][1]
            preprocessed_image = image.reshape(1, 28, 28, 1)
            PREDICTION = model.predict(preprocessed_image)
            predicted_class = PREDICTION.argmax(axis=1)[0]
            predicted_probability = np.max(PREDICTION[0])
            print(f"=======================================> predicted_class : {predicted_class}  |  ground_truth : {ground_truth}  |  predicted_probability : {predicted_probability*100}%  |  augmented: {aug_descriptions[idx]}") 
            if i % num_augs == 0:
                print("\n Augmentation on Image : ", i, "\n")
                j += 1
            if predicted_class == ground_truth:
                correct += 1 
                avg_predicted_probability += predicted_probability
            else:
                incorrect.update({"predicted_class": predicted_class, "ground_truth": ground_truth, " | predicted_probability": predicted_probability})
        accuracy = (correct/total) * 100  
        avg_predicted_probability = (avg_predicted_probability/total) * 100  
        print("=======================================================================================================")
        print("                                           Inference Performance Summary on Augmented Dataset")
        print("Correct: ", correct, "Total: ", total, "Accuracy: ", accuracy, "Avg Prediction Probability: ", avg_predicted_probability)
        print("Incorrect Predictions: ", incorrect) 
        print("=======================================================================================================")
        # print(f"Inference Accuracy: {correct/total*100:.2f}%")

    def render(self, image, title):
        scale_factor = 5 
        if image.ndim == 2:
        # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            # Single-channel image
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Allows the window to be resized
        cv2.resizeWindow(title, new_width, new_height)  # Resize the window to match the image size
        cv2.imshow(title, resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def infer_realtime_old(self, model, test_path):
        for subdir in os.listdir(test_path):
            subdir_path = os.path.join(test_path, subdir)
            for file in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, file)
                if image_path.endswith(('png', 'jpg', 'jpeg')):
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    ", image_path)
                    PREDICTION = self.predict_image(model, image_path)
                    predicted_class = PREDICTION.argmax(axis=1)[0]
                    predicted_probability = np.max(PREDICTION[0])
                    print(f"=====================================>  PREDICTION: [{predicted_class}]  |  PREDICTION PROBABILITY: [{(predicted_probability*100)}%]\n")
def main():
    print("\n       ----------------------------------------- 1. LOADING --------------------------------------\n")
    src_path = '/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/train_dataset/'
    # src_path = '../OCR/segmented_digits/'
    dataset_loader = DatasetLoader()
    custom_images, custom_labels = dataset_loader.load_custom_images(src_path)
    dataset_processor = DatasetProcessor(custom_images, custom_labels) # images, labels
    train_images, val_images, train_labels, val_labels, test_images, test_labels = dataset_processor.split_custom_dataset()
    print(train_images[0], type(train_images[0]))
    # /home/asiadmin/Workspace/CENTRAL_FINAL/trained_models/new_trained_model_v2.h5
    print("Splitted Dataset. Analyzing loaded dataset...\n")
    # unique_labels, class_counts = np.unique(custom_labels, return_counts = True)
    # label_distribution = dict(zip(unique_labels, class_counts))
    # total = np.sum(class_counts)
    # for i in range(len(unique_labels)):
    #     print(f"Class {unique_labels[i]}: {class_counts[i]} instances ({(class_counts[i]/total)*100:.2f}%)")
    # print(f"\nNormalizing Weights...")
    # standardized_weights = (total / class_counts) * len(class_counts)
    # target_weight = np.sum(standardized_weights)
    # normalized_weights = standardized_weights / target_weight 
    # weights = {}
    # for label, weight in zip(unique_labels, normalized_weights):
    #     weights.update({str(label): weight}) 
    #     print(f"Class {label}: {class_counts[label]} instances, Weight: {weight:.4f}\n ")
    # print(f"Unique labels in custom train dataset: {unique_labels}, Label range: {min(unique_labels)} to {max(unique_labels)}")
    # print("weights", weights)
    # print(f'Custom Images Shape: {train_images.shape} Custom Labels Shape: {train_labels.shape}')
    # print(f'Validation Images Shape: {val_images.shape} Validation Labels Shape: {val_labels.shape}')
    # print("\n       ----------------------------------------- 2. CREATING MODEL --------------------------------------\n")
    # model_handler = ModelHandler(input_shape=(28, 28, 1), num_classes=10)
    # model = model_handler.create_model()

    # print("\n       ----------------------------------------- 3. TRAINING / FINE TUNING ON CUSTOM DATASET--------------------------------------\n")
    # model_path = './trained_models/test_inference_model.h5'
    # history = model_handler.train_model(train_images, train_labels, val_images, val_labels, model_path, weights)
    
    print("\n       ----------------------------------------- 4-1. MODEL EVALUATION TEST DATASET--------------------------------------\n")
    trained_model_path = '/home/asiadmin/Workspace/CENTRAL_FINAL/trained_models/test_inference_model.h5'
    trained_model = load_model(trained_model_path)
    evaluator = Evaluator(trained_model, test_images=test_images, test_labels=test_labels)
    # evaluator.evaluate_inference_performance_test_dataset(trained_model, test_images, test_labels)
    print("\n       ----------------------------------------- 4-2. MODEL EVALUATION AUGMENTED DATASET --------------------------------------\n")
    augmented_images = evaluator.augment_images(test_images, test_labels)
    print(len(augmented_images), len(test_images), len(test_labels))
    print(len(augmented_images[0]), len(augmented_images[1]))
    evaluator.evaluate_inference_performance_augmented(trained_model, augmented_images, test_labels)
    # #1. LEARNING CURVE  
    # evaluator.plot_learning_curve(history)
    # model = model_handler.model
    # y_train_pred = np.argmax(model.predict(train_images), axis=1)
    # y_val_pred = np.argmax(model.predict(val_images), axis=1)
    # y_test_pred = np.argmax(model.predict(test_images), axis=1)
    # #2. CLASSIFICATION REPORT 
    # classification_report = evaluator.plot_classification_report(y_test_pred, test_labels)
    # #3. CONFUSION MATRIX
    # confusion_matrix = evaluator.plot_confusion_matrix(y_test_pred, test_labels)
   
if __name__ == "__main__":
    main()

