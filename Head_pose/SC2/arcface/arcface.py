import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(current_dir, '../../')))

import torch
from torchvision import transforms
import PIL
from PIL import Image
from SC2.arcface.networks import Backbone


class ArcFace(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Backbone(50, 0.6, 'ir_se').to(self.device)
        self.model.load_state_dict(torch.load(os.path.realpath(os.path.join(current_dir, 'pretrained/model_ir_se50.pth')), map_location=self.device))
        self.model.eval()

    def embed(self, face: PIL.Image):
        face_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])(face)
        with torch.no_grad():
            emb = self.model(face_tensor.to(self.device).unsqueeze(0))
        return emb