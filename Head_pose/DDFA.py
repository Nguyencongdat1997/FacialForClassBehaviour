import torch 
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import scipy.io as sio
import cv2
import numpy as np

from config import main_dir_path
ddfa_dir_path = main_dir_path + 'ddfa/'
import ddfa.mobilenet_v1 as mobilenet_v1
from ddfa.ddfa_utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from ddfa.ddfa_utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from ddfa.ddfa_utils.cv_plot import plot_pose_box
from ddfa.ddfa_utils.estimate_pose import parse_pose
from ddfa.ddfa_utils.paf import gen_img_paf

#import mtcnn
import sys
mtcnn_dir_path = main_dir_path + 'SC2/'
sys.path.append(mtcnn_dir_path) 
from mtcnn.mtcnn import MTCNN

#import utils
from pose_util import Cooridinates_calculator
cooridinates_calculator = Cooridinates_calculator()


class DDFA_rf():

    def __init__(self, main_dir=ddfa_dir_path):
        '''
          input:
           - main_dir: where to save pretrained models. This directory must have following structure:
                 main_dir
                   |_ models
                   |   |_ phase1_wpdc_vdc.pth.tar
                   |_ train.configs

        '''
        #param:
        self.mode = 'cpu'
        self.dlib_landmark = True
        self.dlib_bbox = False     
        self.bbox_init = 'two'
        self.STD_SIZE = 120    
        self.main_dir = main_dir

        # 1. load pre-tained model               
        self.checkpoint_fp = self.main_dir+ 'models/phase1_wpdc_vdc.pth.tar'
        self.arch = 'mobilenet_1'

        self.checkpoint = torch.load(self.checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        self.model = getattr(mobilenet_v1, self.arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        self.model_dict = self.model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in self.checkpoint.keys():
            self.model_dict[k.replace('module.', '')] = self.checkpoint[k]
        self.model.load_state_dict(self.model_dict)
        if self.mode == 'gpu':
            cudnn.benchmark = True
            self.model = self.model.cuda()
        self.model.eval()

        # 3. load mtcnn model for face detection
        self.mtcnn = MTCNN()   #! TODO: replace by new mtcnn model
        
        # 4. forward
        self.transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    def affine_camera_matrixes_to_vectors(self, matrixes):
        rotation_vectors = []
        translation_vectors = []
        for matrix in matrixes:
            _matrix = np.array(matrix)
            rotation_matrix = _matrix[:3, :3]
            rotation_vector = np.array(cooridinates_calculator.rotationMatrixToEulerAngles(rotation_matrix))
            translation_vector = _matrix[:3, 3]
            rotation_vectors.append(rotation_vector)
            translation_vectors.append(translation_vector)
        return rotation_vectors, translation_vectors

    def detect_pose(self, img_ori): 
        bounding_boxes, pts_res, Ps, poses = self.ddfa_detect_pose(img_ori)
        rotation_vectors, translation_vectors = self.affine_camera_matrixes_to_vectors(Ps)
        return bounding_boxes, rotation_vectors, translation_vectors

    def ddfa_detect_pose(self, img_ori):
        bounding_boxes, _, _ = self.mtcnn.align(img_ori) #TODO: replace when change mtcnn model
        
        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        for rect in  bounding_boxes:            
            # - use detected face bbox
            bbox = [rect[0], rect[1], rect[2], rect[3]]
            roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                if self.mode == 'gpu':
                    input = input.cuda()
                param = self.model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if self.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = self.transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if self.mode == 'gpu':
                        input = input.cuda()
                    param = self.model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)
            
            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)
            #P, pose = parse_pose(param) # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
        
        #print ("3DDFA rotation vector:\n {0}".format(pose))        
        return bounding_boxes, pts_res, Ps, poses
      
    def draw_pose(self, img_ori):
        '''
          input: 
            - image: a cv2 
          output:
            - image with boxes as numpy.ndarray

        '''
        _, pts_res, Ps, poses = self.ddfa_detect_pose(img_ori)
        img_pose = plot_pose_box(img_ori, Ps, pts_res)        
        return img_pose
        
    def draw_pose_image_file(self, data_dir, image_name):
        '''
          input:
            - data_dir: where the image file is saved
            - image_name: name of image file
          output 
            - image with boxes as numpy.ndarray
        '''
        img_fp = data_dir + image_name
        img_ori = cv2.imread(img_fp)             
        return self.draw_pose(img_ori)       


