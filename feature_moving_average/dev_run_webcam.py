import argparse
import logging
import time

import cv2
import numpy as np
import pandas as pd

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

# 座標を取得
# p = 関節の番号
def findPoint(humans, p):
  for human in humans:
    try:
      body_part = human.body_parts[p]
      parts = [0,0]

      # 座標を整数に切り上げで置換
      parts[0] = int(body_part.x * width + 0.5)
      parts[1] = int(body_part.y * height + 0.5)

      # parts = [x座標, y座標]
      return parts
    except:
        pass

# pandasで移動平均を返す
def getMovingAvg(npArray):
  if len(npArray)>0:
    # pandasのdataFrameに埋める
    df = pd.DataFrame(npArray)
    # pandasのrollingメソッドで3区間の移動平均を返す
    return df.rolling(window=3, min_periods=1).mean()
  else:
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)

    # Set the resolution to 640x480
    cam.set(3, 432)  # 3 corresponds to the width
    cam.set(4, 368)  # 4 corresponds to the height

    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # 使用するカメラを指定
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()

    while True:
        ret_val, image = cam.read()

        # 検知された人間
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0),upsample_size=args.resize_out_ratio)

        # 高さと幅を取得
        height,width = image.shape[0],image.shape[1]

        # findPointは上記で記載した座標取得の関数です
        # 各地点の座標
        left_ear = findPoint(humans, 17)
        right_ear = findPoint(humans, 16)

        left_eye = findPoint(humans, 15)
        right_eye = findPoint(humans, 14)

        nose = findPoint(humans, 0)

        if len(left_ear) > 0:
            lEarX_ndarray = np.array([])
            lEarY_ndarray = np.array([])
            lEarX_ndarray.append(left_ear[0])
            lEarY_ndarray.append(left_ear[1])

        if len(right_ear) > 0:
            rEarX_ndarray = np.array([])
            rEarY_ndarray = np.array([])
            rEarX_ndarray.append(right_ear[0])
            rEarY_ndarray.append(right_ear[1])

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()