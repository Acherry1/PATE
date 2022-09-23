# @Author:殷梦晗
# @Time:2022/7/31 9:19

import pandas as pd


def replace_labels():
    df_train_20 = pd.read_excel("f.xlsx")
    df_true = pd.read_excel("d.xlsx")
    # """
    images_filenames = []
    for i in df_train_20["image"]:
        i = i.split("\\")[-2:]
        a = "\\".join(i)
        images_filenames.append(a)
    df_train_20["filenames"] = images_filenames
    df = pd.merge(df_train_20, df_true, on="filenames", how='left')
    # 后面真实操作的时候将true_label改成教师聚合label
    # 使用教师聚合label和加噪后的label
    df = df[["image_x", "true_label"]]
    samples = [list(x) for x in df.values]
    targets = df["true_label"].tolist()
    return samples, targets


if __name__ == '__main__':
    replace_labels()
