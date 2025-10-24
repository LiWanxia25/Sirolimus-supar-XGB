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
model.load_model('model_xgb.json')
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
        
        # 再显示SHAP解释图
        st.subheader("SHAP Explanation")

        print("Model type:", type(model))  # 应该是 <class 'xgboost.sklearn.XGBClassifier'>
        print("Model attributes:", dir(model))  # 检查是否有异常属性
        # 创建SHAP解释器
        explainer_shap = shap.TreeExplainer(model)        

        # 获取SHAP值
        shap_values = explainer_shap.shap_values(pd.DataFrame(final_features_df,columns=feature_names))

        # 确保获取到的shap_values不是None或空值
        if shap_values is None:
            st.error("SHAP values could not be calculated. Please check the model and input data.")
        else:
            # 如果模型返回多个类别的SHAP值（例如分类模型），取相应类别的SHAP值
            if isinstance(shap_values, list):
                shap_values_class = shap_values[0]  # 选择第一个类别的SHAP值
            else:
                shap_values_class = shap_values
   
      # 将标准化前的原始数据存储在变量中
        original_feature_values = pd.DataFrame(features, columns=feature_names)
        # 创建瀑布图
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(values=shap_values_class[0], 
                                               base_values=explainer_shap.expected_value,
                                               data=original_feature_values.iloc[0],
                                               feature_names=original_feature_values.columns.tolist()))
            
        # 调整图表显示
        plt.tight_layout()
        st.pyplot(fig)
        











