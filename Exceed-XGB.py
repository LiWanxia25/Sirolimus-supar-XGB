###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load the model
model = xgb.XGBClassifier()
model.load_model('xgb_model2.json')
scaler = joblib.load('supar_scaler.pkl') 

# Streamlit user interface
st.title("Sirolimus Supra-therapeutic Risk Predictor")

# Define feature names
feature_names = ['Height', 'PLT', 'ALT', 'HDL', 'TC']

# 创建两列布局
col1, col2 = st.columns([1, 1.5])  # 左列稍窄，右列稍宽

with col1:
    # 输入表单
    with st.form("input_form"):
        Height = st.number_input("Height (cm):", min_value=50, max_value=180, value=120)
        PLT = st.number_input("PLT (109/L):", min_value=0, max_value=250, value=10)
        ALT = st.number_input("ALT (U/L):", min_value=0, max_value=120, value=100)
        HDL = st.number_input("HDL (mmol/L):", min_value=0.00, max_value=3.00, value=1.30)
        TC = st.number_input("TC (mmol/L):", min_value=0.00, max_value=10.00, value=4.00)
        submitted = st.form_submit_button("Predict")
        
# 准备输入特征
feature_values = [Height, PLT, ALT, HDL, TC]
features = np.array([feature_values]) 

# 关键修改：使用 pandas DataFrame 来确保列名
features_df = pd.DataFrame(features, columns=feature_names)
standardized_features_1 = scaler.transform(features_df)

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(standardized_features_1, columns=feature_names)

if submitted: 
    with col1:
        # 这里可以留空或放一些其他内容
        pass

    with col2:
        OPTIMAL_THRESHOLD = 0.308
              
        predicted_proba = model.predict_proba(final_features_df)[0]
        prob_class1 = predicted_proba[1]  # 类别1的概率
    
        # 根据最优阈值判断类别
        predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

        # 先显示预测结果
        st.subheader("Prediction Results")

        # 使用更美观的方式显示结果
        if predicted_class == 1:
            st.error(f"Supra-therapeutic Risk: {prob_class1:.1%} (High Risk)")
        else:
            st.success(f"Supra-therapeutic Risk: {prob_class1:.1%} (Low Risk)") 
        
        st.write(f"**Risk Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")

        # 添加分隔线
        st.markdown("---")
        
                # 替换SHAP解释图为特征重要性条形图
        st.subheader("Feature Importance Ranking")
        
        # 获取特征重要性
        importance_scores = model.feature_importances_
        
        # 创建特征重要性条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 排序特征重要性
        indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_scores = [importance_scores[i] for i in indices]
        
        # 绘制水平条形图
        bars = ax.barh(range(len(sorted_features)), sorted_scores)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance Ranking')
        ax.invert_yaxis()  # 最重要的特征在顶部
        
        # 在条形上添加数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)












