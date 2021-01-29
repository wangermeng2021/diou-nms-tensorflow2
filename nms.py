
"""
DIoU proposed in https://arxiv.org/abs/1911.08287v1,
"Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
"""
import numpy as np
import tensorflow as tf

def diou_nms_np(batch_boxes, batch_scores, iou_threshold=0.1, score_threshold=0.1, max_box_num=100):
    """Implementing  diou non-maximum suppression in numpy
     Args:
       batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
       batch_scores:detection scores with shape (N, num_class).
     Returns:
        a list of numpy array: [boxes, scores, classes, num_valid].
     """

    batch_classes = np.argmax(batch_scores, axis=-1)
    batch_scores = np.max(batch_scores, axis=-1)

    batch_size = np.shape(batch_boxes)[0]

    batch_result_boxes = np.empty([batch_size, max_box_num, 4])
    batch_result_scores = np.empty([batch_size, max_box_num])
    batch_result_classes = np.empty([batch_size, max_box_num])
    batch_result_valid = np.empty([batch_size])

    for batch_index in range(batch_size):
        # print(batch_result_boxes[0])
        boxes = batch_boxes[batch_index]
        scores = batch_scores[batch_index]

        classes = batch_classes[batch_index]

        valid_mask = scores > score_threshold

        if np.sum(valid_mask) == 0:
            batch_result_boxes[batch_index] = np.zeros([max_box_num,4])
            batch_result_scores[batch_index] = np.zeros([max_box_num])
            batch_result_classes[batch_index] = np.zeros([max_box_num])
            batch_result_valid[batch_index] = 0
            continue

        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        classes = classes[valid_mask]

        sorted_index = np.argsort(scores)[::-1]
        boxes = boxes[sorted_index]
        scores = scores[sorted_index]
        classes = classes[sorted_index]

        result_boxes = []
        result_scores = []
        result_classes = []
        while boxes.shape[0] > 0:
            result_boxes.append(boxes[0])
            result_scores.append(scores[0])
            result_classes.append(classes[0])
            inter_wh = np.maximum(np.minimum(boxes[0, 2:4], boxes[1:, 2:4])-np.maximum(boxes[0, 0:2], boxes[1:, 0:2]),0)
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]

            box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
            box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

            iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
            center_dist = np.sum(np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                                 axis=-1)
            bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
            diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
            diou = iou_score - center_dist / diagonal_dist
            # print(diou)
            valid_mask = diou <= iou_threshold
            boxes = boxes[1:][valid_mask]
            scores = scores[1:][valid_mask]
            classes = classes[1:][valid_mask]

        num_valid = len(result_boxes)
        num_valid = np.minimum(num_valid, max_box_num)
        result_boxes = np.array(result_boxes)[:num_valid, :]
        result_scores = np.array(result_scores)[:num_valid]
        result_classes = np.array(result_classes)[:num_valid]
        pad_size = max_box_num - num_valid
        result_boxes = np.pad(result_boxes, ((0, pad_size), (0, 0)))
        result_scores = np.pad(result_scores, ((0, pad_size),))
        result_classes = np.pad(result_classes, ((0, pad_size),))

        batch_result_boxes[batch_index] = result_boxes
        batch_result_scores[batch_index] = result_scores
        batch_result_classes[batch_index] = result_classes
        batch_result_valid[batch_index] = num_valid

    return batch_result_boxes,batch_result_scores,batch_result_classes,batch_result_valid
def diou_nms_tf(batch_boxes, batch_scores, iou_threshold=0.01, score_threshold=0.01, max_box_num=100):
    """Implementing  diou non-maximum suppression in tensorflow
     Args:
       batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
       batch_scores:detection scores with shape (N, num_class).
     Returns:
       a list of tensor: [boxes, scores, classes, num_valid].
     """

    batch_classes = tf.math.argmax(batch_scores, axis=-1)
    batch_scores = tf.math.reduce_max(batch_scores, axis=-1)

    batch_result_boxes = []
    batch_result_scores = []
    batch_result_classes = []
    batch_result_valid = []

    for batch_index in tf.range(tf.shape(batch_boxes)[0]):
        boxes = batch_boxes[batch_index]
        scores = batch_scores[batch_index]
        classes = batch_classes[batch_index]

        valid_mask = scores > score_threshold

        if tf.reduce_sum(tf.cast(valid_mask, tf.dtypes.int32)) == 0:
            batch_result_boxes.append(tf.zeros([max_box_num,4]))
            batch_result_scores.append(tf.zeros([max_box_num]))
            batch_result_classes.append(tf.zeros([max_box_num]))
            batch_result_valid.append(tf.constant(0))
            continue
        scores = tf.boolean_mask(scores, valid_mask)
        boxes = tf.boolean_mask(boxes, valid_mask)
        classes = tf.boolean_mask(classes, valid_mask)

        boxes_mask = tf.Variable(tf.zeros((tf.shape(boxes)[0],)),dtype=tf.dtypes.float32,trainable=False)
        boxes_mask = boxes_mask.assign_sub(boxes_mask)

        sorted_index = tf.argsort(scores, direction='DESCENDING')
        scores = tf.gather(scores, sorted_index)
        boxes = tf.gather(boxes, sorted_index)
        classes = tf.gather(classes, sorted_index)

        for boxes_index in tf.range(tf.shape(boxes)[0]-1):
            if boxes_mask[boxes_index] > 0.:
                continue
            inter_wh = tf.maximum(tf.minimum(boxes[boxes_index, 2:4], boxes[boxes_index+1:, 2:4])-tf.maximum(boxes[boxes_index, 0:2], boxes[boxes_index+1:, 0:2]) ,0)
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]

            box1_wh = boxes[boxes_index, 2:4] - boxes[boxes_index, 0:2]
            box2_wh = boxes[boxes_index+1:, 2:4] - boxes[boxes_index+1:, 0:2]

            iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
            center_dist = tf.reduce_sum(tf.square((boxes[boxes_index, 2:4] + boxes[boxes_index, 0:2]) / 2 - (boxes[boxes_index+1:, 2:4] + boxes[boxes_index+1:, 0:2]) / 2),
                                 axis=-1)
            bounding_rect_wh = tf.maximum(boxes[boxes_index, 2:4], boxes[boxes_index+1:, 2:4]) - tf.minimum(boxes[boxes_index, 0:2], boxes[boxes_index+1:, 0:2])
            diagonal_dist = tf.reduce_sum(tf.square(bounding_rect_wh), axis=-1)
            diou = iou_score - center_dist / diagonal_dist
            boxes_mask.assign_add(tf.concat([tf.zeros([boxes_index+1]),tf.cast(diou > iou_threshold, tf.dtypes.float32)],axis=-1))

        result_mask = boxes_mask == 0.
        result_boxes = tf.boolean_mask(boxes, result_mask)
        result_scores = tf.boolean_mask(scores, result_mask)
        result_classes = tf.boolean_mask(classes, result_mask)
        result_valid = tf.shape(result_boxes)[0]
        result_valid = tf.minimum(result_valid,max_box_num)

        result_boxes = result_boxes[:result_valid, :]
        result_scores = result_scores[:result_valid]
        result_classes = result_classes[:result_valid]

        pad_len = max_box_num-result_valid
        result_boxes = tf.pad(result_boxes,((0, pad_len),(0,0)))
        result_scores = tf.pad(result_scores, ((0, pad_len),))
        result_classes = tf.pad(result_classes, ((0, pad_len),))

        batch_result_valid.append(result_valid)
        batch_result_boxes.append(result_boxes)
        batch_result_scores.append(result_scores)
        batch_result_classes.append(result_classes)

    return tf.stack(batch_result_boxes),tf.stack(batch_result_scores),tf.stack(batch_result_classes),tf.stack(batch_result_valid)
