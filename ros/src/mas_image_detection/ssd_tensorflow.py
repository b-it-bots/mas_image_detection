import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# ROS
from rospkg import RosPack
from mas_perception_libs import ImageDetectorBase, ImageDetectionKey


class SSDTfModelsImageDetector(ImageDetectorBase):
    """
    Extension of ImageDetectorBase meant to work with bounding box detection models from 'tensorflow/models' repo
    Largely based on the 'research/object_detection/object_detection_tutorial.ipynb' notebook in the repo
    """
    # Note that 'detection_masks' is skipped since we do not care about segmentation in this case
    _detection_graph = None
    _output_tensor_names = None
    _image_tensor_name = None
    _rp = None
    _conf_threshold = None

    def __init__(self, **kwargs):
        self._rp = RosPack()
        super(SSDTfModelsImageDetector, self).__init__(**kwargs)

    def load_model(self, **kwargs):
        # load confidence threshold
        self._conf_threshold = kwargs.get('conf_threshold', 0.3)

        # load frozen inference graph from
        frozen_graph_package = kwargs.get('frozen_graph_package', None)
        if not frozen_graph_package:
            raise ValueError("'frozen_graph_package' not defined in kwargs file")

        frozen_graph_path = kwargs.get('frozen_graph_path', None)
        if not frozen_graph_path:
            raise ValueError("'frozen_graph_path' not defined in kwargs file")

        pkg_path = self._rp.get_path(frozen_graph_package)
        frozen_graph_full_path = os.path.join(pkg_path, frozen_graph_path)
        if not os.path.exists(frozen_graph_full_path):
            raise ValueError("frozen graph file does not exist: " + frozen_graph_full_path)

        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            graph_def = tf.GraphDef()
            with gfile.FastGFile(frozen_graph_full_path, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

        # load input image and output tensor names
        self._output_tensor_names = kwargs.get('output_tensor_names', None)
        if not self._output_tensor_names:
            raise ValueError("'output_tensor_names' not defined in kwargs file")

        self._image_tensor_name = kwargs.get('image_tensor_name', None)
        if not self._image_tensor_name:
            raise ValueError("'image_tensor_name' is not defined in kwargs file")

    def _detect(self, np_images, orig_img_sizes):
        # stack all images into a single tensor of shape (batch_size, height, width, 3)
        image_array = np.stack(np_images, axis=0)
        predictions = []
        with self._detection_graph.as_default():
            with tf.Session() as sess:
                # get handles for output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                output_tensor_dict = {}
                for key, tensor_name in self._output_tensor_names.iteritems():
                    if tensor_name in all_tensor_names:
                        output_tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                # get image tensor
                image_tensor = tf.get_default_graph().get_tensor_by_name(self._image_tensor_name)

                # run inference
                output_dict = sess.run(output_tensor_dict, feed_dict={image_tensor: image_array})

                # fill detection results
                for img_index, img_size in enumerate(orig_img_sizes):
                    boxes = []
                    num_detection = int(output_dict['num_detections'][img_index])
                    for detection_index in range(num_detection):
                        detection_key = output_dict['detection_classes'][img_index][detection_index]
                        if detection_key not in self._classes:
                            print("WARNING: key '{}' is not in the class dictionary".format(detection_key))
                            continue
                        detected_class = self._classes[detection_key]
                        confidence = output_dict['detection_scores'][img_index][detection_index]
                        if confidence < self._conf_threshold:
                            continue
                        # normalized y_min, x_min, y_max, x_max
                        box = output_dict['detection_boxes'][img_index][detection_index]
                        # convert to true coordinates using img_size
                        box = np.multiply(box, [img_size[1], img_size[0], img_size[1], img_size[0]])
                        box_dict = {
                                ImageDetectionKey.CLASS: detected_class, ImageDetectionKey.CONF: confidence,
                                ImageDetectionKey.X_MIN: box[1], ImageDetectionKey.Y_MIN: box[0],
                                ImageDetectionKey.X_MAX: box[3], ImageDetectionKey.Y_MAX: box[2]
                            }
                        boxes.append(box_dict)
                    predictions.append(boxes)

        return predictions
