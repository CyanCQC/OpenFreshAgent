import os
import json
import numpy as np
from Agent.spectrum.SpectrumModel import SpectralPredictor

model = SpectralPredictor()

def get_spectrum_dict(dir_path):
    # 获取图像文件列表
    img_path = os.path.join(dir_path, "img")
    spec_path = os.path.join(dir_path, "spec")
    img_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 初始化结果字典
    result_dict = {}

    # 遍历每个图像文件
    for img_file in img_files:
        # 提取图像文件的基本名（去掉扩展名）
        base_name = os.path.splitext(img_file)[0]
        ext_name = os.path.basename(img_file)
        # 构建对应的 .xlsx 文件路径
        xlsx_file = os.path.join(spec_path, base_name + '.xlsx')
        # 构造图像文件的完整路径
        img_file_path = os.path.join(img_path, img_file)

        # 检查 .xlsx 文件是否存在
        if os.path.exists(xlsx_file):
            try:
                # 调用 model.predict 获取预测结果
                predictions = model.predict(xlsx_file)
                # 将预测结果存入字典
                result_dict[ext_name] = predictions
                print(f"< {ext_name} > 光谱分析结果：\n {predictions}")
            except Exception as e:
                print(f"# 处理 {xlsx_file} 时出错: {e}")
                result_dict[ext_name] = None  # 出错时标记为 None
        else:
            print(f"# 未找到 {img_file} 对应的 .xlsx 文件")
            result_dict[ext_name] = None  # 未找到文件时标记为 None

    return result_dict