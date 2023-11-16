from typing import Tuple, List, Union
import tensorflow as tf
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from yolonas.config import CONFIG
from yolonas.utils.yolo_utils import decoder, y_pred_from_model_outputs


def custom_yolo_nas_loss(y_true, reg: tf.Tensor, cls: tf.Tensor):
    y_pred = y_pred_from_model_outputs(reg, cls)

    # Extract confidence, coordinates, and class predictions
    pred_confidence = y_pred[:, :, 0]
    pred_boxes = y_pred[:, :, 1:5]
    pred_class_probs = y_pred[:, :, 5:]

    true_boxes = y_true[:, :, :4]
    true_class_probs = y_true[:, :, 4:]

    # Calculate IoU for each pair of predicted and true bounding boxes
    pred_boxes = tf.expand_dims(pred_boxes, axis=2)  # Shape: (batch_size, n_pred_boxes, 1, 4)
    true_boxes = tf.expand_dims(true_boxes, axis=1)  # Shape: (batch_size, 1, n_true_boxes, 4)

    mask = tf.cast(y_true[:, :, 4] != 81, tf.float64)
    mask_expanded = tf.expand_dims(mask, axis=-1)

    regression_loss = tf.keras.losses.Huber()(true_boxes * mask_expanded, pred_boxes * mask_expanded)

    return regression_loss
