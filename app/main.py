from fastapi import FastAPI, Response
import cv2
import numpy as np

import sys
sys.path.append("..")
from run import get_pose_image_from_file

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/get_pose_image/")
async def get_pose_image():
    result_img = get_pose_image_from_file(
        image_path='/Users/kimjichul/Desktop/dev/bodhi/bodhi-backend/images/p1.jpg'
    )  # 원하는 이미지 경로 설정
    _, encoded_image = cv2.imencode('.png', result_img)
    return Response(content=encoded_image.tobytes(), media_type="image/png")