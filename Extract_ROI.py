import cv2
import numpy as np
import json, os 
from PIL import Image  

class Extract_ROI:
    def __init__(self,config_filename):
        self.loadConfig(config_filename)

    def loadConfig(self,config_filename):
        with open(config_filename, 'r') as f:
            self.config = json.load(f)
        # load config

        # debug mode
        self.DEBUG = self.config['DEBUG']

        # initial crop ratio
        self.INITIAL_CROP_RATIO_X = self.config['INITIAL_CROP_RATIO']['X']
        self.INITIAL_CROP_RATIO_Y = self.config['INITIAL_CROP_RATIO']['Y']
        self.INITIAL_CROP_RATIO_WIDTH = self.config['INITIAL_CROP_RATIO']['WIDTH']
        self.INITIAL_CROP_RATIO_HEIGHT = self.config['INITIAL_CROP_RATIO']['HEIGHT']

        # blur
        self.BLUR_KERNEL_SIZE = self.config['BLUR']['KERNEL_SIZE']

        # binary threshold
        self.BINARY_THRESHOLD_ORIGINAL = self.config['BINARY_THRESHOLD']['ORIGINAL']
        self.BINARY_THRESHOLD_MASK = self.config['BINARY_THRESHOLD']['MASK']

        # morphology
        self.MORPHOLOGY_KERNEL_SIZE = self.config['MORPHOLOGY']['KERNEL_SIZE']
        self.MORPHOLOGY_DILATE_ITERATIONS = self.config['MORPHOLOGY']['DILATE_ITERATIONS']
        self.MORPHOLOGY_ERODE_ITERATIONS = self.config['MORPHOLOGY']['ERODE_ITERATIONS']

        # approx polydp
        self.APPROX_POLYDP_EPSILON_CONSTANT = self.config['APPROX_POLYDP']['EPSILON_CONSTANT']
        self.APPROX_DELTA_X_THRESHOLD = self.config['APPROX_POLYDP']['DELTA_X_THRESHOLD']
        self.APPROX_DELTA_Y_THRESHOLD = self.config['APPROX_POLYDP']['DELTA_Y_THRESHOLD']

        # perspective transform
        self.PERSPECTIVE_TRANSFORM_SRC_POINTS_ORDER = self.config['PERSPECTIVE_TRANSFORM']['SRC_POINTS_ORDER']
        self.PERSPECTIVE_TRANSFORM_DST_IMAGE_SIZE = self.config['PERSPECTIVE_TRANSFORM']['DST_IMAGE_SIZE']

        # final roi ratio
        self.FINAL_ROI_RATIO_X_START = self.config['FINAL_ROI_RATIO']['X_START']
        self.FINAL_ROI_RATIO_Y_START = self.config['FINAL_ROI_RATIO']['Y_START']
        self.FINAL_ROI_RATIO_X_END = self.config['FINAL_ROI_RATIO']['X_END']
        self.FINAL_ROI_RATIO_Y_END = self.config['FINAL_ROI_RATIO']['Y_END']
    
    # Method : show image if debug mode is True
    # def showImage(self,window_name,image):
    #     if self.DEBUG:
            # cv2.imshow(window_name,image)
            # cv2.waitKey(0)

    # Method : Load image
    def loadImage(self,openfilename):
        image = cv2.imread(openfilename,cv2.IMREAD_GRAYSCALE)
        # self.showImage('1. original',image)
        return image
    
    # Method : Crop image
    def cropImage(self,image):
        if image is None:
            print("Error: Image is None. Please check if the image was loaded correctly.")
            return None
        height, width = image.shape
        self.height = height
        self.width = width
        x = int(width*self.INITIAL_CROP_RATIO_X)
        y = int(height*self.INITIAL_CROP_RATIO_Y)
        w = int(width*self.INITIAL_CROP_RATIO_WIDTH)
        h = int(height*self.INITIAL_CROP_RATIO_HEIGHT)
        image_crop = image[y:y+h,x:x+w]
        self.image_crop = image_crop
        # self.showImage('2. crop',image_crop)
        return image_crop
    
    # Method : Blur image
    def blurImage(self,image_crop):
        blurred_image = cv2.medianBlur(image_crop, self.BLUR_KERNEL_SIZE)
        # self.showImage('3. blur',blurred_image)
        return blurred_image
    
    # Method : Binary threshold image
    def binaryThresholdImage(self,blurred_image):
        _, binary_image = cv2.threshold(blurred_image, self.BINARY_THRESHOLD_ORIGINAL, 128, cv2.THRESH_BINARY)
        # self.showImage('4. binary',binary_image)
        return binary_image
    
    # Method : Morphology image
    def morphologyImage(self,binary_image):
        kernel = np.ones((self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE), np.uint8)
        binary_morphology_image = cv2.dilate(binary_image, kernel, iterations=1)
        binary_morphology_image = cv2.erode(binary_morphology_image, kernel, iterations=self.MORPHOLOGY_ERODE_ITERATIONS)
        binary_morphology_image = cv2.dilate(binary_morphology_image, kernel, iterations=self.MORPHOLOGY_DILATE_ITERATIONS)
        # self.showImage('5. morphology',binary_morphology_image)
        return binary_morphology_image
    
    # Method : Flood fill image
    def floodFillImage(self,binary_morphology_image):
        h, w = binary_morphology_image.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        filled_image = binary_morphology_image.copy()
        cv2.floodFill(filled_image, mask, (w//2, h//2), 255)
        # self.showImage('6. flood fill',filled_image)
        return filled_image
    
    # Method : Binary threshold image of flood fill image
    def binaryThresholdFilledImage(self,filled_image):
        _, roi_masked_image = cv2.threshold(filled_image, self.BINARY_THRESHOLD_MASK, 255, cv2.THRESH_BINARY)
        # self.showImage('7. binary filled',roi_masked_image)
        return roi_masked_image
    
    # Method : Mask image
    def maskImage(self,image_crop,roi_masked_image):
        image_masked = cv2.bitwise_and(image_crop, image_crop, mask=roi_masked_image)
        # self.showImage('8. mask',image_masked)
        return image_masked
    
    # Method : find countours
    def findContours(self,image_masked):
        contours, _ = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.drawContourImage(contours)
        return contours
    
    # Method : draw contour image if debug mode is True
    def drawContourImage(self,contours):
        if self.DEBUG:
            original_image_copy = self.image_crop.copy()
            contour_image = cv2.drawContours(original_image_copy, contours, -1, (0,255,0), 3)
            # self.showImage('9. contour',contour_image)

    # Method : Approx polydp
    def approxPolydp(self,contour):
        epsilon = self.APPROX_POLYDP_EPSILON_CONSTANT*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        self.drawApproxPolydpImage(approx)
        return approx
    
    # Method : draw approx polydp image if debug mode is True
    def drawApproxPolydpImage(self,approx):
        if self.DEBUG:
            original_image_copy = self.image_crop.copy()
            approx_image = cv2.polylines(original_image_copy, [approx], True, (0,0,255), 3)
            # self.showImage('10. approx',approx_image)
    
    # Method : approx manipulation
    def approxManipulation(self,approx):
        approx_manipulated = np.append( approx, [approx[0]], axis=0 )
        approx_manipulated = approx_manipulated.reshape(-1,2)
        return approx_manipulated
    
    # Method : create approx_delta
    def createApproxDelta(self,approx_manipulated):
        approx_delta = np.diff(approx_manipulated,axis=0)
        return approx_delta
    
    # Method : calculate slope and intercept
    # approx_delta에서 x 또는 y 의 절대값이 특정 값 이상 변화하는 점과 그 직전 점을 잇는 직선의 기울기와 y절편을 계산
    def calculateSlopeIntercept(self,approx_delta,approx_manipulated):
        approx_slope_intercept = []
        for i in range(len(approx_delta)):
            delta_x = approx_delta[i][0]
            delta_y = approx_delta[i][1]
            if abs(delta_x) > self.APPROX_DELTA_X_THRESHOLD or abs(delta_y) > self.APPROX_DELTA_Y_THRESHOLD:
                x1 = approx_manipulated[i][0]
                y1 = approx_manipulated[i][1]
                x2 = approx_manipulated[i+1][0]
                y2 = approx_manipulated[i+1][1]
                slope = (y2 - y1) / (x2 - x1)
                y_intercept = y1 - slope * x1
                approx_slope_intercept.append((slope,y_intercept))
        return approx_slope_intercept
    
    # Method : calculate intersection
    # slope_intercept_tuple_list에 저장된 직선의 기울기와 y절편을 이용하여, 각 직선의 교점 좌표를 계산
    # 여기서 교점은 (x, y) 형태로 저장하고, x,y가 이미지 상의 좌표임
    def calculateIntersection(self,approx_slope_intercept):
        intersection_list = []
        for i in range(len(approx_slope_intercept)):
            for j in range(i+1,len(approx_slope_intercept)):
                slope1, y_intercept1 = approx_slope_intercept[i]
                slope2, y_intercept2 = approx_slope_intercept[j]
                x = (y_intercept2 - y_intercept1) / (slope1 - slope2)
                y = slope1 * x + y_intercept1
                intersection_list.append((x,y))
        
        # intersection_list에서 이미지 밖의 교점을 제거
        intersection_list = [ (int(x),int(y)) for x,y in intersection_list if 0<=x<self.width and 0<=y<self.height ]
        intersection_list = np.array(intersection_list).reshape(-1,1,2).astype(np.int32)
        return intersection_list
    
    # Method : intersection list 정렬
    # PERSPECTIVE_TRANSFORM_SRC_POINTS_ORDER에 맞게 intersection_list를 정렬
    def sortIntersectionList(self,intersection_list):
        leftmost = tuple(intersection_list[intersection_list[:,:, 0].argmin()][0])
        rightmost = tuple(intersection_list[intersection_list[:,:, 0].argmax()][0])
        topmost = tuple(intersection_list[intersection_list[:,:, 1].argmin()][0])
        bottommost = tuple(intersection_list[intersection_list[:,:, 1].argmax()][0])
       
        sorted_points = [ None for _ in range(4) ]
        for idx,value in enumerate(self.PERSPECTIVE_TRANSFORM_SRC_POINTS_ORDER):
            if value=='leftmost':
                sorted_points[idx] = leftmost
            elif value=='rightmost':
                sorted_points[idx] = rightmost
            elif value=='topmost':
                sorted_points[idx] = topmost
            elif value=='bottommost':
                sorted_points[idx] = bottommost
        
        self.drawSortedPoints(sorted_points)
        return sorted_points
    
    # Method : debug모드일 때, sorted_points에 저장된 교점 좌표를 original image에 그리고 교점좌표의 순서 라벨을 표시
    def drawSortedPoints(self,sorted_points):
        if self.DEBUG:
            original_image_copy = self.image_crop.copy()
            for i, point in enumerate(sorted_points):
                cv2.circle(original_image_copy, point, 10, (0,0,255), -1)
                cv2.putText(original_image_copy, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # self.showImage('11. sorted points',original_image_copy)
    
    # Method : perspective transform
    def perspectiveTransform(self,sorted_points):
        w,h = self.PERSPECTIVE_TRANSFORM_DST_IMAGE_SIZE
        pts1 = np.float32(sorted_points)
        pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        transformed_image = cv2.warpPerspective(self.image_crop,M,(w,h))
        # self.showImage('12. perspective transform',transformed_image)
        return transformed_image
    
    # Method : final roi
    def finalROI(self,transformed_image):
        height, width = transformed_image.shape 
        x1 = int(width*self.FINAL_ROI_RATIO_X_START)
        y1 = int(height*self.FINAL_ROI_RATIO_Y_START)
        x2 = int(width*self.FINAL_ROI_RATIO_X_END)
        y2 = int(height*self.FINAL_ROI_RATIO_Y_END)
        roi = transformed_image[y1:y2,x1:x2]
        # self.showImage('13. final roi',roi)
        return roi
    
    # Methos : overall image processing
    # 2024-07-23 14_0 ==========> 2024-07-16 15_00_08__1064725.jpg
    def processImage(self, load_path,save_path, file):
        print("INSIDE PROCESS IMAGE load_path", load_path)
        print("INSIDE PROCESS IMAGE save_path", save_path)
        print("FILE", file)
        # check_path = "/".join(save_path.split("/")[0:-1])
        # if not os.path.exists(check_path):
        #     print(f"  --> Preprocessing Directory for {curr_cam} Does Not Exist")
        #     print(f"  --> Creating {curr_cam} directory: {check_path}\n")
        #     os.makedirs(check_path)
        image = self.loadImage(load_path)
        image_crop = self.cropImage(image)
        blurred_image = self.blurImage(image_crop)
        binary_image = self.binaryThresholdImage(blurred_image)
        binary_morphology_image = self.morphologyImage(binary_image)
        filled_image = self.floodFillImage(binary_morphology_image)
        roi_masked_image = self.binaryThresholdFilledImage(filled_image)
        image_masked = self.maskImage(image_crop,roi_masked_image)
        contours = self.findContours(image_masked)
        approx = self.approxPolydp(contours[0])
        approx_manipulated = self.approxManipulation(approx)
        approx_delta = self.createApproxDelta(approx_manipulated)
        approx_slope_intercept = self.calculateSlopeIntercept(approx_delta,approx_manipulated)
        intersection_list = self.calculateIntersection(approx_slope_intercept)
        sorted_points = self.sortIntersectionList(intersection_list)
        transformed_image = self.perspectiveTransform(sorted_points)
        final_roi_image = self.finalROI(transformed_image)
        save_path = os.path.join(save_path, file)
        print('  --> PREPROCESSED Image {}'.format(save_path))
        cv2.imwrite(save_path, final_roi_image)
        print('  --> SAVED Image {}'.format(save_path))
        image_name = save_path.split("/")[-1]
        if self.DEBUG:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        try: 
            return image_name
        except Exception as e: 
            print(f"Error: No Image Found in the Directory --> Errror: {e}")
            return None 

    
    def processImage2(self, load_path,save_path, curr_cam):
        print("INSIDE PROCESS IMAGE load_path", load_path)
        print("INSIDE PROCESS IMAGE save_path", save_path)
        
        check_path = "/".join(save_path.split("/")[0:-1])
        if not os.path.exists(check_path):
            print(f"  --> Preprocessing Directory for {curr_cam} Does Not Exist")
            print(f"  --> Creating {curr_cam} directory: {check_path}\n")
            os.makedirs(check_path)
        image = self.loadImage(load_path)
        image_crop = self.cropImage(image)
        blurred_image = self.blurImage(image_crop)
        binary_image = self.binaryThresholdImage(blurred_image)
        binary_morphology_image = self.morphologyImage(binary_image)
        filled_image = self.floodFillImage(binary_morphology_image)
        roi_masked_image = self.binaryThresholdFilledImage(filled_image)
        image_masked = self.maskImage(image_crop,roi_masked_image)
        contours = self.findContours(image_masked)
        approx = self.approxPolydp(contours[0])
        approx_manipulated = self.approxManipulation(approx)
        approx_delta = self.createApproxDelta(approx_manipulated)
        approx_slope_intercept = self.calculateSlopeIntercept(approx_delta,approx_manipulated)
        intersection_list = self.calculateIntersection(approx_slope_intercept)
        sorted_points = self.sortIntersectionList(intersection_list)
        transformed_image = self.perspectiveTransform(sorted_points)
        final_roi_image = self.finalROI(transformed_image)
        print('  --> PREPROCESSED Image {}'.format(save_path))
        cv2.imwrite(save_path, final_roi_image)
        print('  --> SAVED Image {}'.format(save_path))
        image_name = save_path.split("/")[-1]
        if self.DEBUG:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        try: 
            return image_name
        except Exception as e: 
            print(f"Error: No Image Found in the Directory --> Errror: {e}")
            return None 
if __name__ == '__main__':


    config_filename = 'config_preprocess.json'
    centralcity_extractroi = Extract_ROI(config_filename)
 