import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title='Predictive Maintenance', layout='wide')
st.title('Predictive Maintenance Dashboard')
st.markdown('XGBoost model with SHAP explainability for industrial equipment failure prediction.')

# Φορτώνουμε από τον φάκελο models/
model = joblib.load('models/xgb_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')
le = joblib.load('models/label_encoder.pkl')
threshold = joblib.load('models/threshold.pkl')

def add_features(df_input):
    df_input = df_input.copy()
    df_input['Power_W'] = df_input['Torque_Nm'] * (df_input['Rotational_speed_rpm'] * 2 * np.pi / 60)
    df_input['Temp_diff'] = df_input['Process_temp_K'] - df_input['Air_temp_K']
    df_input['Energy'] = df_input['Power_W'] * df_input['Tool_wear_min']
    df_input['Torque_x_Toolwear'] = df_input['Torque_Nm'] * df_input['Tool_wear_min']
    df_input['RPM_x_Toolwear'] = df_input['Rotational_speed_rpm'] * df_input['Tool_wear_min']
    df_input['Wear_at_Low_Speed'] = df_input['Tool_wear_min'] * (10000 / df_input['Rotational_speed_rpm'])
    return df_input

tab1, tab2 = st.tabs(['Manual Input', 'Upload Dataset'])

# ─────────────────────────────────────────
# TAB 1: Manual Input
# ─────────────────────────────────────────
with tab1:
    st.subheader('Sensor Values Input')

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.slider('Air Temperature [K]', 295.0, 305.0, 300.0)
        process_temp = st.slider('Process Temperature [K]', 305.0, 315.0, 310.0)
        rpm = st.slider('Rotational Speed [rpm]', 1000, 3000, 1500)

    with col2:
        torque = st.slider('Torque [Nm]', 3.0, 80.0, 40.0)
        tool_wear = st.slider('Tool Wear [min]', 0, 253, 100)
        machine_type = st.selectbox('Machine Type', ['L', 'M', 'H'])

    # Δημιουργούμε DataFrame
    input_data = pd.DataFrame(
        [[air_temp, process_temp, rpm, torque, tool_wear]],
        columns=['Air_temp_K', 'Process_temp_K',
                 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min'])

    input_data = add_features(input_data)
    input_data['Type_encoded'] = le.transform([machine_type])[0]

    # Πρόβλεψη
    proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if proba >= threshold else 0

    st.markdown('---')
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric('Failure Probability', f'{proba:.1%}')
    res_col2.metric('Decision Threshold', f'{threshold:.2f}')

    if prediction == 1:
        res_col3.error('PREDICTION: FAILURE')
    else:
        res_col3.success('PREDICTION: NORMAL')

    # Business impact
    st.markdown('---')
    st.subheader('Cost Estimation')
    cost_col1, cost_col2 = st.columns(2)

    with cost_col1:
        cost_failure = st.number_input(
            'Cost of unplanned failure (€)',
            min_value=0, value=10000, step=1000)
    with cost_col2:
        cost_maintenance = st.number_input(
            'Cost of preventive maintenance (€)',
            min_value=0, value=1000, step=100)

    if prediction == 1:
        saving = cost_failure - cost_maintenance
        st.success(f'Estimated saving from early maintenance: €{saving:,}')
    else:
        st.info('No failure predicted — no maintenance required at this time.')

    # SHAP explanation
    st.markdown('---')
    st.subheader('Decision Analysis (SHAP)')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_data.iloc[0],
        feature_names=feature_names))
    st.pyplot(fig)

# ─────────────────────────────────────────
# TAB 2: Upload Dataset & Retrain
# ─────────────────────────────────────────
with tab2:
    st.subheader('Train with Your Own Data')
    st.markdown('''
    Upload a CSV file with the following columns:
    - `Air temperature [K]`
    - `Process temperature [K]`
    - `Rotational speed [rpm]`
    - `Torque [Nm]`
    - `Tool wear [min]`
    - `Type` (L, M, or H)
    - `Machine failure` (0 or 1)
    ''')

    uploaded_file = st.file_uploader('Select CSV file', type='csv')

    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.write('Data preview:')
        st.dataframe(df_new.head())

        required_cols = ['Air temperature [K]', 'Process temperature [K]',
                        'Rotational speed [rpm]', 'Torque [Nm]',
                        'Tool wear [min]', 'Type', 'Machine failure']
        missing = [c for c in required_cols if c not in df_new.columns]

        if missing:
            st.error(f'Missing columns: {missing}')
        else:
            st.success('Columns are correct!')

            if st.button('Train new model'):
                df_new = df_new.rename(columns={
                    'Air temperature [K]': 'Air_temp_K',
                    'Process temperature [K]': 'Process_temp_K',
                    'Rotational speed [rpm]': 'Rotational_speed_rpm',
                    'Torque [Nm]': 'Torque_Nm',
                    'Tool wear [min]': 'Tool_wear_min'
                })

                df_new = add_features(df_new)
                df_new['Type_encoded'] = le.transform(df_new['Type'])

                X_new = df_new[feature_names]
                y_new = df_new['Machine failure']

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_new, y_new,
                    test_size=0.2,
                    stratify=y_new,
                    random_state=42)

                scale = (y_tr == 0).sum() / (y_tr == 1).sum()

                with st.spinner('Training in progress...'):
                    new_model = XGBClassifier(
                        n_estimators=328,
                        max_depth=5,
                        learning_rate=0.19319689982789898,
                        subsample=0.6263147505489439,
                        colsample_bytree=0.8773375292691852,
                        min_child_weight=1,
                        scale_pos_weight=scale,
                        random_state=42,
                        eval_metric='logloss',
                        verbosity=0
                    )
                    new_model.fit(X_tr, y_tr)

                y_pred_new = (new_model.predict_proba(X_te)[:, 1] >= threshold).astype(int)
                report = classification_report(y_te, y_pred_new, output_dict=True)

                st.subheader('New Model Results')
                m1, m2, m3 = st.columns(3)
                m1.metric('Accuracy', f"{report['accuracy']:.1%}")
                m2.metric('Precision (Failure)', f"{report['1']['precision']:.1%}")
                m3.metric('Recall (Failure)', f"{report['1']['recall']:.1%}")

                joblib.dump(new_model, 'models/xgb_model.pkl')
                st.success('New model saved successfully!')