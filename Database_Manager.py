
import os, shutil
import cv2
class Database_Manager:
    def __init__(self, dir_path, curr_cam, curr_image):
        self.dir_path = dir_path
        self.curr_cam = curr_cam
        self.curr_image = curr_image 
    def clean_digit(self, digit):
        pass 

class File_Manager:
    def __init__(self, dir_path, curr_cam, curr_image):
        self.dir_path = dir_path
        self.curr_cam = curr_cam 
        self.curr_image = curr_image

    def write_ocr_labels_for_evaluation(self, src_path, dest_path, label, probability):
        print("Writing ocr labels for evaluation...\n")
        curr_cam = self.curr_cam
        curr_image = self.curr_image
        sub_dir = os.path.join(dest_path, curr_cam)
        if not os.path.exists(sub_dir):
            print(f"Creating {sub_dir} subdirectory...")
            os.makedirs(sub_dir)
        src_file = os.path.join(src_path, curr_image)
        dest_file = os.path.join(dest_path, curr_cam)
        print("COPY FROM   ", src_file)
        if os.path.isfile(src_file):
            new_file_name = f"[{label}]_[{probability:.2f}]_{curr_image}"
            dest_file = os.path.join(dest_file, new_file_name)
            print("COPY TO    ", dest_file)
            shutil.copy(src_file, dest_file)
        else: 
            print(f"src_file DNE : {src_file}")

    def write_ocr_labels_for_each_digit(self, src_path, dest_path, digit_label, probability_array):
        print("\n--------------------------------------------------------------------------------------------")
        print("Writing ocr labels for training to train_dataset...\n")
        curr_cam = self.curr_cam
        curr_image = self.curr_image
        for i in range(0,10):
            sub_dir = os.path.join(dest_path, str(i))
            if not os.path.exists (sub_dir):
                print(f"Creating Subdirectory for Digit Label {sub_dir}")
                os.makedirs(sub_dir)
        img_name = curr_image.rstrip(".jpg")
        src_path = os.path.join(src_path, curr_image).rstrip(".jpg")
        src_path = os.path.join(src_path, curr_image.rstrip(".jpg")) + "__" 
        for i in range(len(digit_label)):
            new_src_path = src_path + "[" + str(i) + "]" + ".jpg"
            new_file_name = f"[{digit_label[i]}]_[{probability_array[i]:.2f}]_{img_name}__[{digit_label}]__[{curr_cam}][{i}].jpg"
            new_dest_path = os.path.join(dest_path, str(digit_label[i]), new_file_name)
            print(f"SAVING : {new_file_name}..... TO TRAIN_DATASET")
            # print(f"COPY FROM : {new_src_path}")
            shutil.copy(new_src_path, new_dest_path)
            # print(f"COPY TO : {new_dest_path}\n")
        return 0 


    # def write_ocr_labels_for_training(self, src_path, dest_path, label):
    #     # TO DO: 
    #     # 1. SPLIT INTO [0...9]
    #     # 2. TRAIN LOCALLY 
    #     curr_cam = self.curr_cam
    #     curr_image = self.curr_image
    #     sub_dir = os.path.join(dest_path, curr_cam)
    #     if not os.path.exists(sub_dir):
    #         print(f"Creating {sub_dir} subdirectory...")
    #         os.makedirs(sub_dir)
    #     src_file = os.path.join(src_path, curr_image)
    #     dest_file = os.path.join(dest_path, curr_cam)
    #     print("COPY FROM     ", src_file)
    #     if os.path.isfile(src_file):
    #         new_file_name = f"[{label}]_{curr_image}"
    #         dest_file = os.path.join(dest_file, new_file_name)
    #         print("COPY TO    ", dest_file)
    #         shutil.copy(src_file, dest_file)
    #     else: 
    #         print(f"src_file DNE : {src_file}")

        
def main():
    print("Inside Database Manager...")

if __name__ == "__main__":
    main()