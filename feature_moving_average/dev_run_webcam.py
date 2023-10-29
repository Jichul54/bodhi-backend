import argparse
import logging
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import slackweb
slack = slackweb.Slack(
    url="https://hooks.slack.com/services/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")
notify_text = "姿勢を正しましょう"

from find_point import findPoint
from get_moving_avg import getMovingAvg
from visualize_moving_avg import visualizeMovingAvg
from is_good_posture import isGoodPosture

import sys
import os

# 현재 스크립트의 디렉토리를 가져옵니다.
current_directory = os.path.dirname(os.path.abspath(__file__))

# 부모 디렉토리(ildoonet-tf-pose-estimation)를 sys.path에 추가합니다.
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)

# 이제 tf_pose 모듈을 import할 수 있습니다.
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator


logger = logging.getLogger('Analyse Posture Using Moving Average')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' %
                 (args.model, get_graph_path(args.model)))
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

    # Initialize dictionaries for each body part's coordinates
    coordinates = {
        'rEar': [],
        'lEar': [],
        'rEye': [],
        'lEye': [],
        'Nose': []
    }
    
    # Initialize dictionary to store moving average values for each body part as 2D array
    moving_avg_values = {
        'rEar': [],
        'lEar': [],
        'rEye': [],
        'lEye': [],
        'Nose': []
    }

    while True:
        ret_val, image = cam.read()

        # 検知された人間
        humans = e.inference(image, resize_to_default=(
            w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # logger.debug(f"humans: {humans}")
        # 高さと幅を取得
        height, width = image.shape[0], image.shape[1]
        # logger.debug(f"height: {height}")
        # logger.debug(f"width: {width}")

        # findPointは上記で記載した座標取得の関数です
        # 各地点の座標
        left_ear = findPoint(humans, 17, width, height)
        right_ear = findPoint(humans, 16, width, height)
        left_eye = findPoint(humans, 15, width, height)
        right_eye = findPoint(humans, 14, width, height)
        nose = findPoint(humans, 0, width, height)

        # nose 좌표 처리
        if nose is not None and len(nose) > 0:
            coordinates['Nose'].append(nose)

        # left_ear 좌표 처리
        if left_ear is not None and len(left_ear) > 0:
            coordinates['lEar'].append(left_ear)

        # right_ear 좌표 처리
        if right_ear is not None and len(right_ear) > 0:
            coordinates['rEar'].append(right_ear)

        # left_eye 좌표 처리
        if left_eye is not None and len(left_eye) > 0:
            coordinates['lEye'].append(left_eye)

        # right_eye 좌표 처리
        if right_eye is not None and len(right_eye) > 0:
            coordinates['rEye'].append(right_eye)

        for body_part, coords in coordinates.items():
            moving_avg = getMovingAvg(coords)

            if isinstance(moving_avg, bool) and not moving_avg:  # moving_avg가 False인 경우
                continue  # 현재 루프를 스킵하고 다음 루프로 이동

            # X와 Y의 평균값을 각각의 변수에 저장
            x_mean = moving_avg[0].mean()
            y_mean = moving_avg[1].mean()

            # Append the moving average values to the respective list as [x, y]
            moving_avg_values[body_part].append([x_mean, y_mean])

        # 이동평균값 로그 출력
        # for key, value in moving_avg_values.items():
        #     print(f"{key}: {value}")

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()

        for part in moving_avg_values.keys():
            if not moving_avg_values[part]:  # 리스트가 비어있으면 스킵
                continue
            if not isGoodPosture(moving_avg_values[part]):
                print(f"{part} 자세가 무너졌습니다.")
                slack.notify(text=notify_text)
                cv2.destroyAllWindows()
                exit()  # 프로그램 종료

        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

        visualizeMovingAvg(moving_avg_values)

        time.sleep(1)  # 1초 동안 일시 중지
    
    cv2.destroyAllWindows()
