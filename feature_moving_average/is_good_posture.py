def isGoodPosture(points):
    # 임계값(threshold)은 자세가 흐트러졌다고 판단하는 기준이 됩니다. 이 값은 실험적으로 결정하거나 전문가의 조언을 바탕으로 설정하는 것이 좋습니다.
    threshold_x = 100
    threshold_y = 50
    
    # 시작 좌표
    standard_point = points[0]

    # 가장 최근의 좌표만을 고려하여 변화량 계산
    latest_point = points[-1]
    delta_x = abs(latest_point[0] - standard_point[0])
    delta_y = abs(latest_point[1] - standard_point[1])

    # 변화량이 임계값을 초과하는지 확인
    if delta_x > threshold_x or delta_y > threshold_y:
        return False
    return True
