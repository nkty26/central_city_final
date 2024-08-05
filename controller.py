from ocr_module import OCR 
import os
import json
import shutil
from acquire_data import acquire_data
from ocr_module import OCR 
from Extract_ROI import Extract_ROI
from inference_module import Inference, DatasetLoader, DatasetSplitter, ModelHandler
from Database_Manager import Database_Manager, File_Manager
import numpy as np
from keras.api.models import load_model

def write_ocr_results(data, write_path):
    for camera in data:
        sub_dir = os.path.join(write_path, camera['camera_name'])
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        camera_name = camera['camera_name']
        dest_dir = './DATASETS/ocr_results/'
        dest_dir = os.path.join(dest_dir, camera_name)
        print("------------------------- Writing OCR Results For : ", dest_dir, "-------------------------")
        for i in range(len(camera['camera_data'])):
            predicted_labels = camera['camera_data'][i]['prediction']
            image_path = camera['camera_data'][i]['image_path']
            image_name = camera['camera_data'][i]['curr_image']
            src_file = image_path.replace("segmented", "preprocessed") + ".jpg"
            print("COPY FROM     ", src_file)
            if os.path.isfile(src_file):
                new_file_name = f"{predicted_labels}_{image_name}" + ".jpg"
                dest_file = os.path.join(dest_dir, new_file_name)
                print("COPY TO    ", dest_file)
                shutil.copy(src_file, dest_file)
            print("PREDICTIONS", predicted_labels, "\n")

def main():
    # TRAIN LOCALLY AND SAVE MODEL TO "./trained_models"
    model_path = '/home/asiadmin/Workspace/CENTRAL_FINAL/trained_models/new_trained_model_v2.h5'
    trained_model = load_model(model_path)
    inference_module = Inference() 
    print("\n\n======================================== 1. Data Acquisition From IP Camera...  ========================================")
    acquisition_results = acquire_data() #1. Data Acquisition from IP Camera 
    print("Acquired Data: ", acquisition_results)
    config_preprocess_json = '/home/asiadmin/Workspace/CENTRAL_FINAL/config_preprocess.json'
    # preprocess_results = '/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/'
    extract_roi = Extract_ROI(config_preprocess_json)
    camera_length = len(acquisition_results)
    OCR_RESULTS =[[] for _ in range(camera_length)] 
    
    for i in range(camera_length):
        try: 
            curr_cam = acquisition_results[i]['curr_cam']
            load_path = acquisition_results[i]['acquisition_path']
            image_name = "".join(load_path.split("/")[-1:])
            print(curr_cam)
        except Exception as e: 
            print(f"no CAM_{i+1} initialized during data acquisition...")
            continue 
        if image_name: 
            print(f"\n======================================== 2. Preprocessing Image with ROI for [{curr_cam}][{image_name}]...  ========================================")
            save_path = os.path.join('/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/preprocessed_images', curr_cam, image_name)
            preprocessed = extract_roi.processImage(load_path, save_path, curr_cam)
            print(f"\n======================================== 3. Executing ROI Digit Segmentation for [{curr_cam}][{image_name}]...  ========================================")
            seg_load_path = os.path.join('/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/preprocessed_images', curr_cam)
            seg_save_path =  os.path.join('/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/segmented_digits', curr_cam)
            ocr = OCR()
            seg_results = ocr.segment_roi(seg_load_path, image_name, curr_cam)
            inference_src_path = ocr.save_segmentation_data(seg_results, seg_save_path, curr_cam, image_name)
            curr_subdir = inference_src_path.split("/")[-1]
            img_idx = 0 
            curr_image_list = [] 
            for img in os.listdir(inference_src_path): 
                if image_name.rstrip('.jpg') in img:  
                    img_idx+=1 
                    curr_image_list.append(img)
            sorted_image_list = sorted(curr_image_list, key = lambda x: x.split('[')[1][0])
            OCR_PREDICTION = ""
            PROBABILITY_ARRAY = []
            for current_image in sorted_image_list:
                curr_img_path = os.path.join('/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/segmented_digits/', curr_cam, curr_subdir, current_image)
                index = current_image.split('[')[1][0]
                PREDICTIONS = inference_module.infer_realtime(trained_model, curr_img_path, current_image)
                PREDICTION = PREDICTIONS[0]
                PREDICTION_PROB = PREDICTIONS[1]
                OCR_PREDICTION+=str(PREDICTION)
                PROBABILITY_ARRAY.append(PREDICTION_PROB)
                OCR_RESULTS[i].append({"prediction": int(PREDICTION), "probability": round(float(PREDICTION_PROB),2), "image_path": curr_img_path, "image_name": current_image, "digit_index": index})
            OCR_RESULTS = json.dumps(OCR_RESULTS, indent = 4 )
            print(OCR_RESULTS)
            OVERALL_ACCURACY = round((np.sum(PROBABILITY_ARRAY) / 7 * 100),2)
            print("\n\n==========================================================================================================")
            print(f"\n                                OCR PREDICTION FOR {image_name} ======>   [{OCR_PREDICTION}][{OVERALL_ACCURACY}%]")
            print("\n==========================================================================================================")
            file_manager = File_Manager(seg_load_path, curr_cam, image_name)
            dest_path =  '/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/ocr_results_with_labels/'
            file_manager.write_ocr_labels_for_evaluation(seg_load_path, dest_path, OCR_PREDICTION, OVERALL_ACCURACY)
            digit_src_path =  '/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/segmented_digits/'
            digit_dest_path =  '/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/train_dataset/'
            file_manager.write_ocr_labels_for_each_digit(seg_save_path, digit_dest_path, OCR_PREDICTION, PROBABILITY_ARRAY)
            # ocr.save_segmentation_data_all_images(seg_results, seg_load_path, seg_save_path, curr_cam, i)
            # print(segmentation_results)
if __name__ == '__main__':
    main()
