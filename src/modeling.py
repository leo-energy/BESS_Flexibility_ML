# --- 脱敏版核心代码：BESS Flexibility ML Pipeline ---
# 对应仓库：BESS_Flexibility_ML/src/modeling.py
# 功能：数据加载 → 特征工程 → RF训练 → SHAP解释

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import shap

# 1. 数据加载与同步（ENTSO-E UTC对齐）
def load_entsoe_data(market_files):
    """
    加载并同步ENTSO-E多市场小时级数据。
    将所有市场时间戳对齐至UTC标准。
    """
    df_list = []
    for market, path in market_files.items():
        temp_df = pd.read_csv(path)
        temp_df.iloc[:, 0] = pd.to_datetime(temp_df.iloc[:, 0])
        temp_df['Actual_Time'] = temp_df.iloc[:, 0] + pd.to_timedelta(temp_df['Position'] - 1, unit='h')
        temp_df.set_index('Actual_Time', inplace=True)
        temp_df = temp_df[['Price']]
        temp_df.columns = [f'Price_{market}']
        df_list.append(temp_df)
    
    df = pd.concat(df_list, axis=1)
    df = df.interpolate(method='linear')
    return df

# 2. 特征工程（滞后项 + 时间特征）
def create_features(df):
    """构造滞后价格特征和时序特征。"""
    model_df = df.copy()
    
    # 滞后特征（1小时和24小时）
    for m in ['NL', 'ES', 'PT']:
        model_df[f'Lag1_{m}'] = model_df[f'Price_{m}'].shift(1)
        model_df[f'Lag24_{m}'] = model_df[f'Price_{m}'].shift(24)
    
    # 时间特征
    model_df['hour'] = model_df.index.hour
    model_df['dayofweek'] = model_df.index.dayofweek
    
    return model_df.dropna()

# 3. 模型训练（随机森林）
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} EUR/MWh")
    return rf, X_test, y_test, y_pred

# 4. SHAP可解释性分析
def run_shap_analysis(rf_model, X_sample):
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)
    return shap_values

# 5. 主流程（示例）
if __name__ == "__main__":
    # 假设数据文件路径
    files = {
        'NL': 'data/NL_prices.csv',
        'ES': 'data/ES_prices.csv',
        'PT': 'data/PT_prices.csv'
    }
    df = load_entsoe_data(files)
    df_feat = create_features(df)
    
    X = df_feat.drop(['Price_NL', 'Price_ES', 'Price_PT'], axis=1)
    y = df_feat['Price_NL']
    
    rf_model, X_test, y_test, y_pred = train_random_forest(X, y)
    
    # SHAP分析（取前500样本加速）
    run_shap_analysis(rf_model, X_test.iloc[:500])
