import cv2
import numpy as np
from PIL import Image

#import mtcnn
import sys
from config import main_dir_path
mtcnn_dir_path = main_dir_path + 'SC2/'
sys.path.append(mtcnn_dir_path) 
from mtcnn.mtcnn import MTCNN


class OpenCV5landmarks:
  
    def __init__(self):
        #initialize params
        self.mtcnn = MTCNN() #TODO: replace when update new mtcnn model              
        
        # 3D model points.
        self.model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip                            
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner

                                ])
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion    
        self.camera_matrix = None        
        
    def detect_pose(self, image):        
        """
            This function detect pose in an image, return list of rotation_vectors and translation_vectors
            Inp:
                - image: nparray
            Out:
                - bounding_boxes
                - rotation_vectors
                - translation_vectors 

        """
        size = image.shape         
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        self.camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
        #print ("Camera Matrix :\n {0}".format(camera_matrix))        

        bounding_boxes, landmarks, _ = self.mtcnn.align(image)
        print('Num bouding box: ', len(bounding_boxes))
        #print('Num bouding box: ', len(bounding_boxes))
        if len(landmarks) < 1:
            print('mtcn detect '+str(len(landmarks))+' landmark.')
            return []
                  
        rotation_vectors = []
        translation_vectors = []
        
        for i in range(len(landmarks)):                 
            #2D image points. If you change the image, you need to change vector
            image_points = np.array([
                                        (landmarks[i][2], landmarks[i][7]),     # Nose tip                           
                                        (landmarks[i][0], landmarks[i][5]),     # Left eye left corner
                                        (landmarks[i][1], landmarks[i][6]),     # Right eye right corne
                                        (landmarks[i][3], landmarks[i][8]),     # Left Mouth corner
                                        (landmarks[i][4], landmarks[i][9])      # Right mouth corner
                                    ], dtype="double")

            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            # if success:                
            #     rotation_vectors.append(rotation_vector)
            #     translation_vectors.append(translation_vector)
            # else:
            #     rotation_vectors.append(None)
            #     translation_vectors.append(None)
            rotation_vectors.append(rotation_vector)
            translation_vectors.append(translation_vector)

        return bounding_boxes, rotation_vectors, translation_vectors