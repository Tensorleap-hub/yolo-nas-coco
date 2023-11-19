from copy import deepcopy
from typing import Union, List, Dict

import numpy as np
import tensorflow as tf
from code_loader.contract.responsedataclasses import BoundingBox
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy._typing import NDArray
from yolonas.config import CONFIG
from yolonas.utils.yolo_utils import decoder


def get_predict_bbox_list(reg_fixed: tf.Tensor, cls: tf.Tensor) -> List[BoundingBox]:
    """
    Description: This function takes a TensorFlow tensor data as input and returns a list of bounding boxes representing predicted annotations.
    Input: data (tf.Tensor): A TensorFlow tensor representing the output data.
    Output: bb_object (List[BoundingBox]): A list of bounding box objects representing the predicted annotations.
    """
    outputs = decoder(loc_data=[np.expand_dims(reg_fixed, 0)], conf_data=[np.expand_dims(cls, 0)], prior_data=[None],
                      from_logits=False, decoded=True)
    bb_object = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=CONFIG['BACKGROUND_LABEL'])
    return bb_object


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = np.array(bb_array)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = bb_array[i][1:5]
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=str(bb_array[i][min(5, len(bb_array[i]) - 1)])) #FIXME add categories

            bb_list.append(curr_bb)
    return bb_list


def polygon_to_bbox(vertices):
    """
    Converts a polygon representation to a bounding box representation.

    Args:
        vertices (list): List of vertices defining the polygon. The vertices should be in the form [x1, y1, x2, y2, ...].

    Returns:
        list: Bounding box representation of the polygon in the form [x, y, width, height].

    Note:
        - The input list of vertices should contain x and y coordinates in alternating order.
        - The function calculates the minimum and maximum values of the x and y coordinates to determine the bounding box.
        - The bounding box representation is returned as [x, y, width, height], where (x, y) represents the center point of the
          bounding box, and width and height denote the size of the bounding box.
    """

    xs = [x for i, x in enumerate(vertices) if i % 2 == 0]
    ys = [x for i, x in enumerate(vertices) if i % 2 != 0]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    # Bounding box representation: (x, y, width, height)
    bbox = [(min_x + max_x) / 2., (min_y + max_y) / 2., max_x - min_x, max_y - min_y]

    return bbox


def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: int) -> np.ndarray:
    """
    Calculates the Intersection over Union (IOU) for all pairs of bounding boxes.

    This function utilizes vectorization to efficiently compute the IOU for all possible pairs of bounding boxes.
    By leveraging NumPy's array operations, the calculations are performed in parallel, leading to improved performance.

    Args:
        bboxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h].
        image_size (int): Size of the image.

    Returns:
        np.ndarray: Array containing the IOU values for all pairs of bounding boxes.
    """

    # Reformat all bboxes to (x_min, y_min, x_max, y_max)
    bboxes = np.asarray([bbox[:-1] for bbox in bboxes]) * image_size
    num_bboxes = len(bboxes)
    # Calculate coordinates for all pairs
    x_min = np.maximum(bboxes[:, 0][:, np.newaxis], bboxes[:, 0])
    y_min = np.maximum(bboxes[:, 1][:, np.newaxis], bboxes[:, 1])
    x_max = np.minimum(bboxes[:, 2][:, np.newaxis], bboxes[:, 2])
    y_max = np.minimum(bboxes[:, 3][:, np.newaxis], bboxes[:, 3])

    # Calculate areas for all pairs
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Calculate intersection area for all pairs
    intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

    # Calculate union area for all pairs
    union_area = areas[:, np.newaxis] + areas - intersection_area

    # Calculate IOU for all pairs
    iou = intersection_area / union_area
    iou = iou[np.triu_indices(num_bboxes, k=1)]
    return iou


def count_obj_bbox_occlusions(img: np.ndarray, bboxes: np.ndarray, occlusion_threshold: float, calc_avg_flag: bool) -> \
        Union[float, int]:
    """
    Counts the occluded bounding boxes of a specific object category in an image.

    This function takes an image and an array of bounding boxes as input and counts the number of occluded
    bounding boxes of a specific object category. The occlusion is determined based on the Intersection over Union (IOU)
    between the bounding boxes.

    Args:
        img (np.ndarray): Image represented as a NumPy array.
        bboxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h, label].
        occlusion_threshold (float): Threshold value for determining occlusion based on IOU.
        calc_avg_flag (bool): Flag indicating whether to calculate the average occlusion count.

    Returns:
        Union[float, int]: Number of occluded bounding boxes of the specified object category.
                           If calc_avg_flag is True, it returns the average occlusion count as a float.
                           If calc_avg_flag is False, it returns the total occlusion count as an integer.

    """
    img_size = img.shape[0]
    label = CONFIG["CATEGORIES"].index('object')
    obj_bbox = bboxes[bboxes[..., -1] == label]
    if len(obj_bbox) == 0:
        return 0.0
    else:
        ious = calculate_iou_all_pairs(obj_bbox, img_size)
        occlusion_count = len(ious[ious > occlusion_threshold])
        if calc_avg_flag:
            return int(occlusion_count / len(obj_bbox))
        else:
            return occlusion_count


def remove_label_from_bbs(bbs_object_array, removal_label, add_to_label):
    new_bb_arr = []
    for bb in bbs_object_array:
        if bb.label != removal_label:
            new_bb = deepcopy(bb)
            new_bb.label = new_bb.label + "_" + add_to_label
            new_bb_arr.append(new_bb)
    return new_bb_arr


def calculate_overlap(box1, box2):
    # Extract coordinates of the bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate the overlap area
    overlap_area = w_intersection * h_intersection

    return overlap_area


def extract_and_cache_bboxes(idx: int, data: Dict):
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    bboxes = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], 5])
    max_anns = min(CONFIG['MAX_BB_PER_IMAGE'], len(anns))
    for i in range(max_anns):
        ann = anns[i]
        if isinstance(ann['bbox'], list):
            img_size = (x['height'], x['width'])
            class_id = ann['category_id']
            bbox = np.expand_dims(ann['bbox'], 0)[0].astype(np.float32)
            bbox /= np.array((img_size[1], img_size[0], img_size[1], img_size[0])).astype(np.float32)
            bboxes[i, :4] = bbox
            bboxes[i, 4] = 0
    bboxes[max_anns:, 4] = CONFIG['BACKGROUND_LABEL']
    return bboxes


def draw_image_with_boxes(image, bounding_boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
        confidence, label = bbox.confidence, bbox.label

        # Convert relative coordinates to absolute coordinates
        abs_x = x * image.shape[1]
        abs_y = y * image.shape[0]
        abs_width = width * image.shape[1]
        abs_height = height * image.shape[0]

        # Create a rectangle patch
        rect = patches.Rectangle(
            (abs_x - abs_width / 2, abs_y - abs_height / 2),
            abs_width, abs_height,
            linewidth=2, edgecolor='r', facecolor='none'
        )

        # Add the rectangle to the axes
        ax.add_patch(rect)

        # Display label and confidence
        ax.text(abs_x - abs_width / 2, abs_y - abs_height / 2 - 5,
                f"{label} {confidence:.2f}", color='r', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # Show the image with bounding boxes
    plt.show()