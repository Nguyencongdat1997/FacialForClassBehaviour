from PIL import Image
import torch
import torch.nn.functional as F
import os
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage.transform import resize
import numpy as np

from models.vgg import VGG
from models.resnet import ResNet18

from config import main_dir_path
#import mtcnn
import sys
mtcnn_dir_path = main_dir_path + 'SC2/'
sys.path.append(mtcnn_dir_path) 
from mtcnn.mtcnn import MTCNN


class EmotionDetector:
    def __init__(self, model='VGG19', main_dir=main_dir_path, face_detector='undefined', use_cuda = False, reliability = 0.8):
        self.main_dir = main_dir
        self.face_detector = face_detector
        self.use_cuda = use_cuda
        self.reliability = reliability
        self.cut_size = 44
    
        self.transform_test = transforms.Compose([
            transforms.TenCrop(self.cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        if model == 'VGG19':
            self.net = VGG('VGG19')
        elif model == 'Resnet18':
            self.net = ResNet18()
        self.checkpoint = torch.load(os.path.join(self.main_dir + 'pretrained_model/' + model, 'PrivateTest_model.t7'), map_location='cpu')
        self.net.load_state_dict(self.checkpoint['net'])
        if self.use_cuda:
            self.net.cuda()
        self.net.eval()

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def detect_emotion_single_face(self, raw_img):       
        '''
            This function is used to dectect facial emotion for an image of single face
        '''
        gray = self.rgb2gray(raw_img)
        gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = self.transform_test(img)             
              
        ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if self.use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = self.net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)
        if torch.max(score) > self.reliability:
            #return score, predicted
            return score, self.class_names[int(predicted.cpu().numpy())]
        else:
            return score, 'UNK'
      
    def detect_emotion_multiple_face(self, raw_img):     
        '''
            This function is used to dectect facial emotion for an image with multiple faces
        '''
        
        if isinstance(self.face_detector, MTCNN):                           
            bounding_boxes, _, _ = self.face_detector.align(raw_img)
        else:
            print('No MTCNN face dectector found.') #TODO: change to add more facedetection model to do experiments)
        
        scores = []
        predicteds = []
        for facebox in bounding_boxes:
            face_img = raw_img[int(facebox[1]): int(facebox[3]),
                         int(facebox[0]): int(facebox[2])]
      
            gray = self.rgb2gray(face_img)
            gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

            img = gray[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            inputs = self.transform_test(img)             

            ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if self.use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs, volatile=True)
            outputs = self.net(inputs)

            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)
            
            scores.append(score)
            #predicteds.append(predicted)            
            if torch.max(score) > self.reliability:
                predicteds.append(self.class_names[int(predicted.cpu().numpy())])
            else:
                predicteds.append('UNK')
        
        return bounding_boxes, scores, predicteds

    def detect_emotion_from_faceboxes(self, faceboxes):     
        '''
            
        '''        
        scores = []
        predicteds = []
        for facebox in faceboxes:      
            gray = self.rgb2gray(face_img)
            gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

            img = gray[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            inputs = self.transform_test(img)             

            ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if self.use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs, volatile=True)
            outputs = self.net(inputs)

            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)
            
            scores.append(score)
            #predicteds.append(predicted)            
            if torch.max(score) > self.reliability:
                predicteds.append(self.class_names[int(predicted.cpu().numpy())])
            else:
                predicteds.append('UNK')
        
        return scores, predicteds