from ctypes.wintypes import PFLOAT
from fileinput import filename
from pydoc import describe

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import scipy.optimize as optimize
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
import itertools
import os

plt.rcParams['font.family'] = "MS Gothic"
plt.rcParams["font.size"] = 20
plt.tight_layout()

timeList = ["起床","夕食","入浴","寝る準備","就寝"] #時刻データの列名
timeList_added = []#追加する時刻データの列名
wakeup_value = ["A","B","C","D","E"] # おきっぷし評価
pd.set_option('display.max_columns', 10)
df_common = None

def func_1d(x, a, b):
    return a * x + b

def convert_to_timedelta(time_str):
    hours, minutes = map(int, time_str.split(":"))
    return timedelta(hours=hours, minutes=minutes)

def timedelta_to_string(td_str):
    # print(type(td_str))
    if not isinstance(td_str, timedelta):
        # print("Ret!!!")
        return td_str
    else:
        days = td_str / timedelta(days=1)
        hours = td_str / timedelta(hours=1)
        minutes = td_str / timedelta(minutes=1)
        seconds = td_str / timedelta(seconds=1)
        milliseconds = td_str / timedelta(milliseconds=1)
        res = None
        if(hours >= 0):
            res = f"{int(hours):02}:{int(minutes % 60):02}:{int(seconds % 60):02}.{int(milliseconds % 1000):04}"
        else:
            res = f"{int(hours):03}:{int(minutes % 60):02}:{int(seconds % 60):02}.{int(milliseconds % 1000):04}"
        # print(res)
        return res

def count_format(count_datestr):
    if isinstance(count_datestr, (int, float, np.int64)):  # 数値の場合の処理
        return int(count_datestr)  # 単純に整数として返す
    elif isinstance(count_datestr, str):  # 文字列の場合の処理
        count_datetime = dt.strptime(count_datestr, '%H:%M:%S.%f')
        count_raw = count_datetime.hour * 60 + count_datetime.minute
        return int(count_raw)
    else:  # それ以外の場合はエラーを投げる
        raise ValueError(f"Unsupported type for count_format: {type(count_datestr)}")

# Excelのシリアル値をPythonのdatetimeに変換する関数
def excel_to_datetime(excel_serial):
    base_date = dt(1899, 12, 30)  # Excelのシリアル値基準
    return base_date + timedelta(days=excel_serial)

# 軸ラベルを時刻表記に変更
def format_func(value, tick_number):
    index = int(value)
    datetime_values = excel_to_datetime(value)

    if 0 <= index < len(datetime_values):
        return datetime_values[index].strftime("%H:%M")
    return ""

def format_timedelta_as_hhmm(x, _):
    td = pd.Timedelta(nanoseconds=x)  # xはナノ秒なので秒に変換
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours:02}:{minutes:02}"  # hh:mm形式

def calc_score(x,y):
    x_sec = x.apply(lambda td: td.total_seconds() if isinstance(td, pd.Timedelta) else td)
    y_sec = y.apply(lambda td: td.total_seconds() if isinstance(td, pd.Timedelta) else td)
    popt, _ = optimize.curve_fit(func_1d, x_sec, y_sec)
    r2 = metrics.r2_score(y_sec, func_1d(x_sec, *popt))

    return popt,r2

def analyze(elms,df,title=""):
    fig, ax = plt.subplots(figsize=(20, 10))
    df.plot(ax=ax,x="日付")
    ax.set_title(f"生活リズム_{title}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_timedelta_as_hhmm))
    os.makedirs(f"pics/{time_now}", exist_ok=True)
    os.makedirs(f"pics/{time_now}/全体図", exist_ok=True)
    os.makedirs(f"pics/{time_now}/詳細", exist_ok=True)
    fig.savefig(f"pics/{time_now}/全体図/{title}.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 12))
    for elm in elms:

        print(elm)
        filename = f"{elm[0]}_{elm[1]}"
        x = df[elm[0]]
        y = df[elm[1]]
        # print(x)
        # 近似式の分析
        popt,r2 = calc_score(x,y)
        res = x.corr(y)
        # print(res,popt,r2)
        # プロット
        # print(x)
        ax.set_title(f"生活リズムの関係_{elm[0]}と{elm[1]}")
        ax.scatter(x, y)
        ax.text(0.05, 0.9,f"corr={res:.2f},y={popt[0]:.3f}x+{popt[1]:.3f},R2={r2:.3f}", ha="left", va="top", transform=ax.transAxes)
        ax.set_xlabel(elm[0])
        ax.set_ylabel(elm[1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_timedelta_as_hhmm))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_timedelta_as_hhmm))
        # plt.show()
        # 保存
        fig.savefig(f"pics/{time_now}/{filename}.png")
        # プロットを消去
        ax.cla()

        for wv in wakeup_value:
            df_targ = df_common[df_common["起きっぷり"]==wv]
            df_targ = df[df["日付"].isin(df_targ["日付"])]
            # print(df_targ)
            x = df_targ[elm[0]]
            y = df_targ[elm[1]]
            popt,r2 = calc_score(x,y)
            res = x.corr(y)
            ax.scatter(x, y,label=f"起きっぷり:{wv}:corr={res:.2f},y={popt[0]:.3f}x+{popt[1]:.3f},R2={r2:.3f}")
        plt.legend()
        ax.set_xlabel(elm[0])
        ax.set_ylabel(elm[1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_timedelta_as_hhmm))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_timedelta_as_hhmm))
        ax.set_title(f"生活リズムの関係(起きっぷり別)_{elm[0]}と{elm[1]}")
        # 保存と消去
        # plt.show()
        fig.savefig(f"pics/{time_now}/詳細/{filename}_detail.png")
        ax.cla()
    plt.close(fig)

def analyze_rolling(df,days,title):
    fig, ax = plt.subplots(figsize=(20, 10))
    rolling_df = df.loc[:,"日付"]
    # print(df)
    df = df.iloc[:,1:]
    for col in df.columns:
        tmp_df = df[col]
        # 秒単位で数値リストに変換
        data_in_seconds = tmp_df.apply(lambda td: td.total_seconds())

        # Rolling 計算を適用
        tmp_df = data_in_seconds.rolling(window=days, min_periods=1).mean()
        # ナノ秒単位を timedelta に変換
        tmp_df = tmp_df.apply(lambda x: timedelta(seconds=x))  # timedelta を秒から生成
        # print(tmp_df)
        # exit(0)
        rolling_df = pd.concat([rolling_df,tmp_df],axis=1)
    # プロット
    rolling_df.plot(ax=ax,x="日付")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_timedelta_as_hhmm))
    # print(rolling_df)
    ax.set_title(f"生活リズム(移動平均{days}日)_{title}")
    # plt.show()
    fig.savefig(f"pics/{time_now}/全体図/移動平均{days}_{title}.png")
    plt.close(fig)

if __name__ == "__main__":
    # 識別用に今の時間を取得
    time_now = dt.now().strftime("%Y%m%d_%H%M%S")
    # データを入れたパスを指定
    path = "./Timememo.xlsx"
    df = pd.read_excel(path, header=0)
    # print(df)

    df = df.dropna(how="any")
    df_common = df.iloc[:,:3]
    columns_as_lists = {col: df[col].tolist() for col in df.columns}
    # print(df)
    # print(df.dtypes)

    # 日時を変換
    df["日付"] = pd.to_datetime(df["日付"], format="%Y-%m-%d")

    df_Show = df.copy()
    for tL in timeList:
        # print(tL)
        # Timedelta形式に変換後、文字列として表示
        df_Show[tL] = df_Show[tL].apply(lambda x: pd.to_timedelta(x) if isinstance(x, str) else x)
        df_Show[tL] = df_Show[tL].apply(lambda x: timedelta_to_string(x))

    print("DF_Show")
    print(df_Show)
    # print(df_Show.dtypes)

    # Describeの結果をタイムデータに変換して表示
    describe_df = df.describe()

    for tL in timeList:
        describe_df[tL] = describe_df[tL].apply(lambda x: pd.to_timedelta(x, unit='m'))
        describe_df[tL] = describe_df[tL].apply(lambda x: timedelta_to_string(x))
    describe_df.loc['count'] = describe_df.loc['count'].apply(lambda x: count_format(x))
    print("Describe_DF")
    print(describe_df)
    # print(describe_df.dtypes)
    addelms = itertools.combinations(timeList[1:], 2) #二つの組み合わせとして列挙
    # データ群を追加
    moredata_df = df.loc[:,"日付"]
    for elm in addelms:
        print(elm)
        title = f"{elm[0]}から{elm[1]}"
        timeList_added.append(title)
        data = df.loc[:,elm[1]] - df.loc[:,elm[0]]
        adddata_df = pd.DataFrame(data=data,columns=[title])
        moredata_df = pd.concat([moredata_df,adddata_df],axis=1)


    analyze_origin_df = df.drop(columns=["用事","起きっぷり"])
    # print(moredata_df)
    # ここから解析
    elms = itertools.combinations(timeList, 2) #二つの組み合わせとして列挙
    elms_more = itertools.combinations(timeList_added,2)

    analyze(elms,df=analyze_origin_df,title="元データ")
    analyze(elms_more,df=moredata_df,title="追加データ")
    for day in range(2,15):
        analyze_rolling(analyze_origin_df,days=day,title="元データ")
        analyze_rolling(moredata_df, days=day,title="追加データ")

