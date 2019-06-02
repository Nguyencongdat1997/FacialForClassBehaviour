import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(current_dir, '../../')))

import cv2
import PIL
from PIL import Image
import numpy as np


def cv_to_pil_image(image: np.ndarray):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = Image.fromarray(image)
    return image


def pil_to_cv_image(image: PIL.Image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image
