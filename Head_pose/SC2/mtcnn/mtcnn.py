import sys
import os
from typing import List, Tuple, Any

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.realpath(current_dir))
sys.path.append(os.path.realpath(os.path.join(current_dir, '../../')))

from mtcnn_detector import MtcnnDetector
import mxnet as mx
import numpy as np
from skimage.transform import SimilarityTransform
import cv2


class MTCNN(object):
    def __init__(self, min_face_size: float = 20., factor: float = 0.709, threshold: List[float] = [0.6, 0.7, 0.8]):
        self.device = mx.cpu() if len(mx.test_utils.list_gpus()) == 0 else mx.gpu(0)
        self.min_face_size = min_face_size
        self.factor = factor
        self.threshold = threshold
        self.model_path = os.path.realpath(os.path.join(current_dir, 'mtcnn_model'))
        self.model = MtcnnDetector(model_folder=self.model_path, ctx=self.device, num_worker=1, accurate_landmark = True, threshold=self.threshold, minsize=self.min_face_size, factor=self.factor)

    def align(self, image: np.ndarray) -> Tuple[List[Any], List[Any], List[Any]]:
        ret = self.model.detect_face(image, det_type=0)
        if ret is None:
            return [], [], []
        bounding_boxes, landmarks = ret
        if bounding_boxes.shape[0] == 0:
            return [], [], []
        reference_facial_points = np.array([
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156]
        ], dtype=np.float32)
        reference_facial_points[:, 0] += 8.
        transform = SimilarityTransform()
        faces = []
        for landmark in landmarks:
            tmp_landmark = np.array(landmark, dtype=np.float32).reshape((2, 5)).T
            transform.estimate(tmp_landmark, reference_facial_points)
            M = transform.params[0:2, :]
            warped_face = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
            faces.append(warped_face)
        return bounding_boxes, landmarks, faces
