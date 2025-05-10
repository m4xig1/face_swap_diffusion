import os
from PIL import Image
from src.id_utils.insightface_package import FaceAnalysis2, analyze_faces
from diffusers.utils import load_image
import numpy as np
from tqdm import tqdm

import os
from copy import copy

class Aligner():
    def __init__(self):
        self.face_detector = FaceAnalysis2(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, input_id_images):
        cropped_images = []
        bboxes = []
        embeds = []
        for orig_img in input_id_images:
            img = np.array(copy(orig_img))
            img = img[:, :, ::-1]
            faces = analyze_faces(self.face_detector, img)
            if len(faces) == 0:
                print('No face')
                continue

            orig_arr = np.array(orig_img)
            faces = sorted(faces, key=lambda x: -(x['bbox'][3] - x['bbox'][1]) * (x['bbox'][2] - x['bbox'][0]))
            bbox = faces[0]['bbox'].astype(np.int32)
            embeds.append(faces[0]['embedding'])

            bbox = np.clip(bbox, 0, max(orig_img.size))
            croped = orig_arr[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            cropped_img = Image.fromarray(croped)
            cropped_images.append(cropped_img)
            bboxes.append(bbox.tolist())
        
        return cropped_images, bboxes, embeds