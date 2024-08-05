# ocr_module.py
import os
import cv2
import numpy as np
class OCR:
    def __init__(self ):
        pass

    def clean_digit(self, digit):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_digit = cv2.morphologyEx(digit, cv2.MORPH_OPEN, kernel)
        h, w = cleaned_digit.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_margin = 5
        mask[mask_margin:h-mask_margin, mask_margin:w-mask_margin] = 1
        cleaned_digit = cv2.bitwise_and(cleaned_digit, cleaned_digit, mask=mask)
        return cleaned_digit
    
    def segment_roi(self, load_path, image_name, curr_cam):
        digit_images = [] 
        filename = os.path.join(load_path, image_name) 
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(load_path, filename) 
            image_path = filename
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            print("Image Height: ", height, "  |  Image Width: ", width)
            if image is None:
                print(f"Error: Could not load image at {image_path}")
            crop_sequence = [] 
            crop_sequence = np.array([70, 145, 222, 300, 370, 450, 524])
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, bin_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
            for i in range(7):  # 7 digits
                if i == 0: 
                    x_start = 10         
                else: 
                    x_start = x_end
                x_end = crop_sequence[i]
                digit_crop = cleaned[:, x_start:x_end]
                # digit_images.append(digit_crop)
                digit_images.append({
                        'digit_index': i,
                        'filename': filename,
                        'coordinates': (0,0,0,0),
                        'image': digit_crop,
                        'curr_cam': curr_cam
                })
        return digit_images
    
    
    # def segment_roi(self, load_path, image_name, curr_cam):
    #     digits = []
    #     filename = os.path.join(load_path, image_name)
    #     if filename.endswith(('.jpg', '.jpeg', '.png')):
    #         image_path = os.path.join(load_path, filename)
    #         image = cv2.imread(image_path)
    #         if image is None:
    #             print(f"Error: Could not load image at {image_path}")
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         _, bin_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #         cleaned = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
    #         contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         bboxes = [cv2.boundingRect(contour) for contour in contours]
    #         sorted_bboxes = sorted(bboxes, key=lambda x: x[0])
    #         try:
    #             x_min = 0
    #             x_max = max([x + w for x, _, w, _ in sorted_bboxes])
    #             y_min = min([y for _, y, _, _ in sorted_bboxes])
    #             y_max = max([y + h for _, y, _, h in sorted_bboxes])
    #         except Exception as e:
    #             print(f"Error: {e}")
    #         print(f"BBOX COORDINATES : x_min {x_min} | x_max {x_max} | y_min {y_min} | y_max {y_max} | BBOX Length {len(sorted_bboxes)}\n")
    #         (height, width) = cleaned.shape[:2]
    #         print("Image Height: ", height, "  |  Image Width: ", width)
    #         x_stride = int((x_max - x_min) / 7)
    #         y_start = y_min
    #         y_end = y_max
    #         x_start = x_min
    #         x_end = x_start + x_stride
    #         for j in range(len(bboxes)): 
    #             if x_end > width:
    #                 x_end = width  
    #             if x_max - x_end < x_stride or x_end > width:
    #                 x_end = width 
    #             new_bbox = cleaned[y_start:y_end, x_start:x_end]
    #             if new_bbox.size > 0 and x_end <= width and y_end <= height:
    #                 print(f"   index: {j} ----> y_start: {y_start}, y_end: {y_end}, x_start: {x_start}, x_end: {x_end}")
    #                 new_bbox = self.clean_digit(new_bbox)
    #                 digit_image = new_bbox
    #                 digits.append({
    #                     'digit_index': j,
    #                     'filename': filename,
    #                     'coordinates': (x_start, x_end, y_start, y_end),
    #                     'image': digit_image,
    #                     'curr_cam': curr_cam
    #                 })
    #                 x_start = x_end
    #                 x_end = x_start + x_stride
    #             else:
    #                 break
    #         print("\n")
    #     return digits


    def save_segmentation_data(self, data, output_path, curr_cam, image_name): 
        save_path = os.path.join(output_path[0:-5], curr_cam, image_name[0:-4])
        if not os.path.exists(output_path):
            print(f"--> Segmentation Directory for {curr_cam} Does Not Exist. Creating {curr_cam} directory: ", output_path)
            os.makedirs(output_path) 
        if not os.path.exists(save_path):
            print(f"Creating subdirectory... [{image_name[0:-4]}]\n")
            os.makedirs(save_path)
        for i in range(len(data)): # numdigits 
            digit = data[i]
            fname = image_name[0:-4]
            filename = f"{fname}__[{i}].jpg"
            bgr_image = cv2.cvtColor(digit['image'], cv2.COLOR_GRAY2BGR)
            temp_save_path = os.path.join(save_path, filename)
            if cv2.imwrite(temp_save_path, bgr_image):
                print(f" --> SUCCESSFULLY SAVED {filename} ==> {temp_save_path}")
            else:
                print(f"Failed to save {filename}")
        print(f"***** Completed Saving Segmented Digits for {filename}...\n")
        return save_path
