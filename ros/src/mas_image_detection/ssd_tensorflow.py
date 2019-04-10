import numpy as np
from mas_perception_libs import ImageDetectorBase, ImageDetectionKey


class SSDTensorflowImageDetector(ImageDetectorBase):
    def __init__(self, **kwargs):
        super(SSDTensorflowImageDetector, self).__init__(**kwargs)

    def load_model(self, **kwargs):
        pass

    def _detect(self, np_images, orig_img_sizes):
        predictions = []
        return predictions
