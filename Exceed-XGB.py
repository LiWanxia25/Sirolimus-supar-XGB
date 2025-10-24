###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
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
col1, col2 = st.columns([1, 1.5])

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
features_df = pd.DataFrame(features, columns=feature_names)
standardized_features_1 = scaler.transform(features_df)
final_features_df = pd.DataFrame(standardized_features_1, columns=feature_names)

if submitted: 
    with col1:
        pass

    with col2:
        OPTIMAL_THRESHOLD = 0.308
              
        predicted_proba = model.predict_proba(final_features_df)[0]
        prob_class1 = predicted_proba[1]
        predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

        # 显示预测结果
        st.subheader("Prediction Results")
        if predicted_class == 1:
            st.error(f"Supra-therapeutic Risk: {prob_class1:.1%} (High Risk)")
        else:
            st.success(f"Supra-therapeutic Risk: {prob_class1:.1%} (Low Risk)") 
        
        st.write(f"**Risk Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")
        st.markdown("---")
        
        # 特征重要性分析 - 完全移除SHAP相关代码
        st.subheader("Feature Importance Ranking")
        
        try:
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
            plt.close()
            
        except Exception as e:
            st.error(f"特征重要性分析失败: {e}")
            # 如果特征重要性也失败，显示简单的特征表格
            st.info("显示输入特征值:")
            feature_table = pd.DataFrame({
                'Feature': feature_names,
                'Your Value': feature_values
            })
            st.dataframe(feature_table)



