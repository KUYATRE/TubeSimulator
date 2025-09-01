import pandas as pd
import yaml

def sort_dataframe_by_column(df: pd.DataFrame):
    mv_columns = [f"MV_H{i}" for i in range(1, 7)]

    for col in mv_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame")

    new_df = df[mv_columns].copy()

    return new_df

def sumarize_mv_activity(df: pd.DataFrame):
    mv_columns = [f"MV_H{i}" for i in range(1, 7)]

    for col in mv_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame")

    df["Active_Heater"] = df[mv_columns].sum(axis=1)

    summary = df["Active_Heater"].value_counts().reset_index()
    summary.columns = ["Active_Heater", "Duration(sec)"]

    full_range = pd.DataFrame({"Active_Heater": range(0, 7)})
    summary = full_range.merge(summary, on="Active_Heater", how="left").fillna(0)
    summary["Duration(sec)"] = summary["Duration(sec)"].astype(int)

    return summary

def calc_avg(df: pd.DataFrame, heater_power: int):
    sum_val = 0
    sum_time = 0

    for i in range(1,7):
        sum_val = sum_val + df["Active_Heater"][i] * df["Duration(sec)"][i]
        sum_time = sum_time + df["Duration(sec)"][i]

    return heater_power * sum_val / sum_time

def load_heater_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg



if __name__ == "__main__":
    data = {
    'MV_H1': [1, 1, 0, 0, 1, 1, 1],
    'MV_H2': [1, 1, 0, 0, 0, 1, 1],
    'MV_H3': [0, 1, 0, 0, 0, 1, 1],
    'MV_H4': [0, 0, 0, 0, 0, 1, 1],
    'MV_H5': [0, 0, 0, 0, 0, 0, 1],
    'MV_H6': [0, 0, 0, 0, 0, 0, 1],
    }
    df = pd.DataFrame(data)
    summary = sumarize_mv_activity(df)
    print(summary)