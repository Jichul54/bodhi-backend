import pandas as pd

# pandasで移動平均を返す
def getMovingAvg(npArray):
  if len(npArray)>0:
    # pandasのdataFrameに埋める
    df = pd.DataFrame(npArray)
    # pandasのrollingメソッドで3区間の移動平均を返す
    return df.rolling(window=3, min_periods=1).mean()
  else:
    return False