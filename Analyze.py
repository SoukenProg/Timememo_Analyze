from ctypes.wintypes import PFLOAT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

timeList = ["起床","夕食","入浴","寝る準備","就寝"] #時刻データの列名

def convert_to_timedelta(time_str):
    hours, minutes = map(int, time_str.split(":"))
    return timedelta(hours=hours, minutes=minutes)

if __name__ == "__main__":
    #データを入れたパスを指定
    path = "./Timememo.csv"
    df = pd.read_csv(path)

    print(df)
    print(df.dtypes)

    #日時を変換
    df["日付"]=pd.to_datetime(df["日付"],format="%Y-%m-%d")

    for tList in timeList:
        df[tList] = df[tList].apply(convert_to_timedelta)

    print(df)
    print(df.dtypes)
