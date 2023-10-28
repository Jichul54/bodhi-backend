# 座標を取得
# p = 関節の番号
def findPoint(humans, p, w, h):
  for human in humans:
    try:
      body_part = human.body_parts[p]
      parts = [0,0]

      # 座標を整数に切り上げで置換
      parts[0] = int(body_part.x * w + 0.5)
      parts[1] = int(body_part.y * h + 0.5)

      # parts = [x座標, y座標]
      return parts
    except:
        pass