from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import json

from PIL import Image
import io

import sys
sys.path.append('..') 
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from feature_moving_average.find_point import findPoint
from feature_moving_average.get_moving_avg import getMovingAvg
from feature_moving_average.is_good_posture import isGoodPosture

app = FastAPI()

# 모든 출처를 허용하는 CORS 미들웨어를 추가합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인을 허용합니다. 실제 배포시에는 정확한 도메인을 명시하는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드를 허용합니다.
    allow_headers=["*"],  # 모든 HTTP 헤더를 허용합니다.
)

# OpenPose 모델 설정
model = 'mobilenet_thin'
w, h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

@app.post("/analyze_posture/")
async def analyze_image(file: UploadFile = File(...), data: str = Form(...)):
    # JSON 데이터를 파싱합니다.
    json_data = json.loads(data)
    coordinates = json_data.get('coordinates', {})
    moving_avg_values = json_data.get('moving_avg_values', {})
    
    # 이미지 파일을 읽고 처리합니다.
    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # OpenPose로 이미지 분석
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    # Get the height and width of the image
    height, width = image.shape[0], image.shape[1]

    # Extract keypoints using OpenPose
    body_parts = ['Nose', 'lEar', 'rEar', 'lEye', 'rEye']
    body_part_indices = [0, 17, 16, 15, 14]
    for part, idx in zip(body_parts, body_part_indices):
        point = findPoint(humans, idx, width, height)
        if point is not None and len(point) > 0:
            coordinates[part] = coordinates.get(part, []) + [point]
    
    # Calculate moving averages
    for body_part, coords in coordinates.items():
        moving_avg = getMovingAvg(coords)

        if isinstance(moving_avg, bool) and not moving_avg:  # moving_avg가 False인 경우
            continue  # 현재 루프를 스킵하고 다음 루프로 이동

        # X와 Y의 평균값을 각각의 변수에 저장
        x_mean = round(moving_avg[0].mean(), 1)
        y_mean = round(moving_avg[1].mean(), 1)

        # Append the moving average values to the respective list as [x, y]
        moving_avg_values[body_part].append([x_mean, y_mean])
        
    # 자세 평가 로직
    posture_evaluation_result = True  # 기본적으로 자세가 좋다고 가정
    for part in moving_avg_values.keys():
        print(f"{part} moving averages: {moving_avg_values[part]}")
        if not moving_avg_values[part]:  # 리스트가 비어있으면 스킵
            continue
        if not isGoodPosture(moving_avg_values[part]):
            posture_evaluation_result = False
            break
    
    return JSONResponse(content={
        "coordinates": coordinates,
        "moving_avg_values": moving_avg_values,
        "posture_evaluation": posture_evaluation_result
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)