import requests
import numpy as np
import matplotlib.pyplot as plt

# API 服务器地址
# API_URL = "http://35.196.45.78:8000/predict/"

API_URL = "https://color-prediction-492048110470.us-central1.run.app/predict/"


# 真实测试颜色（手动设置的 ground truth）
true_rgb = (235, 189, 190)  # 2001-4B

# 读取测试图片
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
image_path = os.path.join(script_dir, "test.jpg")  # 生成绝对路径

print("Using Image Path:", image_path)

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(API_URL, files=files)


# 解析 API 响应
if response.status_code == 200:
    result = response.json()
    captured_color = tuple(result["captured_color"])  # API 返回的拍摄颜色
    predicted_color = tuple(result["predicted_color"])  # API 预测的真实颜色

    print("\nResults:")
    print("拍摄到的目标色 (Cp):", captured_color)
    print("模型实际上的真实色：", true_rgb)
    print("模型预测的真实色 (Cs_pred):", predicted_color)

    # **可视化颜色**
    plt.figure(figsize=(10, 4))

    # **拍摄颜色**
    plt.subplot(1, 3, 1)
    color_display = np.full((100, 100, 3), captured_color, dtype=np.uint8)
    plt.imshow(color_display)
    plt.title(f'Captured Color\nRGB{captured_color}')
    plt.axis('off')

    # **预测颜色**
    plt.subplot(1, 3, 2)
    color_display = np.full((100, 100, 3), predicted_color, dtype=np.uint8)
    plt.imshow(color_display)
    plt.title(f'Predicted Color\nRGB{predicted_color}')
    plt.axis('off')

    # **真实颜色**
    plt.subplot(1, 3, 3)
    color_display = np.full((100, 100, 3), true_rgb, dtype=np.uint8)
    plt.imshow(color_display)
    plt.title(f'True Color\nRGB{true_rgb}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print("❌ API 请求失败，状态码:", response.status_code)
    print("❌ 错误信息:", response.text)
