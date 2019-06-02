import cv2
import numpy as np
from PIL import Image
import math

class Cooridinates_calculator:

    def __init__ (self):
        """
        """

    def rotation_vector_to_pose_vector_in_planes(self, rotation_vector, max_length):
        pose_vector_3D = np.dot(self.eulerAnglesToRotationMatrix(rotation_vector), np.array([0,max_length,0]))
        pose_vector_XY = pose_vector_3D[:2]
        pose_vector_XZ = pose_vector_3D[::2]
        return pose_vector_XY, pose_vector_XZ

    def isRotationMatrix(self, R) :
        # Checks if a matrix is a valid rotation matrix.
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R) :
        # Calculates rotation matrix to euler angles
        # The result is the same as MATLAB except the order
        # of the euler angles ( x and z are swapped ).
        
        assert not self.isRotationMatrix(R), 'The rotation vector is not valid'
         
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
         
        singular = sy < 1e-6
     
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
     
        return [x, y, z]

    def eulerAnglesToRotationMatrix(self, theta) :
        theta = np.array(theta)
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])                                       
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])                 
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])                                         
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R


class Visualizer:

    def __init__ (self):
        """
            
        """
        self.cooridinates_calculator = Cooridinates_calculator()
       
    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 128, 128), line_width=2):
        """
            This function draw a 3D box as annotation of pose into image
        """
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_annotation_boxes(self, image, rotation_vectors, translation_vectors, color=(255, 128, 128), line_width=2):
        """
            This function draw 3D boxes into image
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
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion    

        for i in range(len(rotation_vectors)): 
            # if rotation_vectors[i] == None or translation_vectors[i] == None:
            #     continue           
            self.draw_annotation_box(image, rotation_vectors[i], translation_vectors[i], color=color)

    def draw_text_information(self, image, bounding_boxes, rotation_vectors, translation_vectors):
        """
            This function draw lines of text describing information
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        for i in range(len(bounding_boxes)):
            # if rotation_vectors[i] == None or translation_vectors[i] == None:
            #     continue
            txt = 'R: ' + np.array2string(rotation_vectors[i].flatten(), separator=',' ) #+ ' T: ' + np.array2string(translation_vectors[i].flatten(), separator=',' )
            cv2.putText(image, str(txt), (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), font, 0.25, (40, 214, 63), 1, cv2.LINE_AA)

    def detect_pose_in_image(self, image, head_pose_detector):
        """
            This function detect pose in an image, then draw box to visualize
            Inp:            
                - head_pose_detector: a variable of OpenCV5landmarks//...
                - image: input image as a nparray
            Out:
                - an image as nparray 

        """        
        tm = cv2.TickMeter()
        tm.start()
        bounding_boxes, rotation_vectors, translation_vectors = head_pose_detector.detect_pose(image)
        tm.stop()
        print('Processed time: ', tm.getTimeSec())

        #visualize
        self.draw_annotation_boxes(image, rotation_vectors, translation_vectors)
        self.draw_text_information(image, bounding_boxes, rotation_vectors, translation_vectors)

        return image   

    def draw_axis(self, img, pose, tdx=None, tdy=None, vector_length = 50):    
        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.size[:2]
            tdx = width / 2
            tdy = height / 2
            
        R = eulerAnglesToRotationMatrix(pose)       #Cooridinates_calculator.eulerAnglesToRotationMatrix
        X = np.dot(np.array([1,0,0]), R)
        Y = np.dot(np.array([0,1,0]), R)
        Z = np.dot(np.array([0,0,-1]), R)
        #print(X,Y,Z)
        draw = ImageDraw.Draw(img) 
        draw.line((int(tdx), int(tdy), int(X[0]*vector_length) + tdx, int(X[1]*vector_length) + tdy),(0,0,255),2)
        draw.line((int(tdx), int(tdy), int(Y[0]*vector_length) + tdx, int(Y[1]*vector_length) + tdy),(0,255,0),2)
        draw.line((int(tdx), int(tdy), int(Z[0]*vector_length) + tdx, int(Z[1]*vector_length) + tdy),(255,0,0),2) 

        return img

    def draw_rotation_arrows(self, image, bounding_boxes, rotation_vectors, maxlength):
        for i in range(len(bounding_boxes)):
            pose_vector_XY, pose_vector_XZ = self.cooridinates_calculator.rotation_vector_to_pose_vector_in_planes(rotation_vectors[i], maxlength)        
            cv2.arrowedLine(image, 
                            ( int(bounding_boxes[i][0]), int(bounding_boxes[i][1]) ),
                            ( int(bounding_boxes[i][0]+pose_vector_XZ[0]), int(bounding_boxes[i][1]+pose_vector_XZ[1]) ),
                            color = (0,0,255),
                            thickness = 2,
                           )
            cv2.arrowedLine(image, 
                            ( int(bounding_boxes[i][2]), int(bounding_boxes[i][1]) ),
                            ( int(bounding_boxes[i][2]+pose_vector_XY[0]), int(bounding_boxes[i][1]+pose_vector_XY[1]) ),
                            color = (0,255,0),
                            thickness = 2,
                           )