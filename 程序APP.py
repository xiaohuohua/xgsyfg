import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的 XGBoost 模型
# 确保你的目录下有 'XGBoost.pkl' 文件
model = joblib.load('XGBoost.pkl')

# 特征范围定义
feature_ranges = {
    "NtproBNP": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
    "BMI": {"type": "numerical", "min": 10.000, "max": 50.000, "default": 24.555},
    "LeftAtrialDiam": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 3.7},
    "AFCourse": {"type": "numerical", "min": 0, "max": 100, "default": 12},
    "AtrialFibrillationType": {"type": "categorical", "options": [0, 1], "default": 0},
    "SystolicBP": {"type": "numerical", "min": 50, "max": 200, "default": 116},
    "EF": {"type": "numerical", "min": 18, "max": 100, "default": 71},
    "Pulse": {"type": "numerical", "min": 0, "max": 1000, "default": 24},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # ------------------ 修改部分开始 ------------------
    # 计算 SHAP 值
    # 1. 构造 DataFrame 以保留特征名称，这对于 SHAP 图的可读性很重要
    feature_names = list(feature_ranges.keys())
    X_df = pd.DataFrame([feature_values], columns=feature_names)

    # 2. 初始化 Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # 3. 适配 XGBoost 的输出格式
    # XGBoost 二分类通常返回单个数组，而不是列表。
    # 这里做一个判断：如果是列表（兼容旧代码或多分类），取对应类别；
    # 如果不是列表（XGBoost二分类标准输出），直接使用。
    if isinstance(shap_values, list):
        shap_val_to_plot = shap_values[predicted_class]
        base_value = explainer.expected_value[predicted_class]
    else:
        # XGBoost 二分类情况
        shap_val_to_plot = shap_values
        base_value = explainer.expected_value

    # 4. 生成 SHAP 力图
    shap_fig = shap.force_plot(
        base_value,                 # 期望值 (Base Value)
        shap_val_to_plot,           # SHAP 值
        X_df,                       # 特征值 (DataFrame格式)
        matplotlib=True,            # 必须开启，才能使用 plt 保存
        show=False                  # 禁止自动弹窗，防止阻塞 Streamlit
    )
    # ------------------ 修改部分结束 ------------------

    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
