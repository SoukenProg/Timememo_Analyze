from ctypes.wintypes import PFLOAT
from pydoc import describe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

timeList = ["起床","夕食","入浴","寝る準備","就寝"] #時刻データの列名
pd.set_option('display.max_columns', 10)
def convert_to_timedelta(time_str):
    hours, minutes = map(int, time_str.split(":"))
    return timedelta(hours=hours, minutes=minutes)

def timedelta_to_string(td_str):
    print(type(td_str))
    if not isinstance(td_str, timedelta):
        print("Ret!!!")
        return td_str
    else:
        days = td_str / timedelta(days=1)
        hours = td_str / timedelta(hours=1)
        minutes = td_str / timedelta(minutes=1)
        seconds = td_str / timedelta(seconds=1)
        milliseconds = td_str / timedelta(milliseconds=1)

        res = f"{int(hours):02}:{int(minutes % 60):02}:{int(seconds % 60):02}.{int(milliseconds % 1000):04}"
        print(res)
        return res


if __name__ == "__main__":
    # データを入れたパスを指定
    path = "./Timememo.xlsx"
    df = pd.read_excel(path, header=0)
    df = df.dropna(how="any")
    columns_as_lists = {col: df[col].tolist() for col in df.columns}
    print(df)
    print(df.dtypes)

    # 日時を変換
    df["日付"] = pd.to_datetime(df["日付"], format="%Y-%m-%d")

    df_Show = df.copy()
    for tL in timeList:
        print(tL)
        # Timedelta形式に変換後、文字列として表示
        df_Show[tL] = df_Show[tL].apply(lambda x: pd.to_timedelta(x) if isinstance(x, str) else x)
        df_Show[tL] = df_Show[tL].apply(lambda x: timedelta_to_string(x))

    print("DF_Show")
    print(df_Show)
    print(df_Show.dtypes)

    # Describeの結果をタイムデータに変換して表示
    describe_df = df.describe()
    for tL in timeList:
        describe_df[tL] = describe_df[tL].apply(lambda x: pd.to_timedelta(x, unit='m'))
        describe_df[tL] = describe_df[tL].apply(lambda x: timedelta_to_string(x))

    print("Describe_DF")
    print(describe_df)
    print(describe_df.dtypes)