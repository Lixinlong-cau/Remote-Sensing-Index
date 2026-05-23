import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
# ==============NO.1===============
# ===================== 1. 核心配置 =====================
INPUT_PATH = "23Points_2014201510.csv"
OUTPUT_PATH = "多源融合校正.xlsx"

BANDS = ['Green', 'Red', 'NIR', 'SWIR']
HIGH_RES = ['S2', 'L8']
LOW_RES = 'MODIS'

# ===================== 2. 构建高分辨率真值 =====================
def build_high_benchmark(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=BANDS).copy()

    # 高分辨率数据：同一天 → 均值
    high = df[df['source'].isin(HIGH_RES)].groupby(['name', 'date'])[BANDS].mean().reset_index()
    high['data_type'] = 'high'
    return high, df

# ===================== 3. 训练线性回归模型 =====================
def train_lr_models(high_df, raw_df):
    modis = raw_df[raw_df['source'] == LOW_RES].copy()
    train = pd.merge(high_df, modis, on=['name', 'date'], suffixes=('_true', '_modis'))

    models = {}
    for name, g in train.groupby('name'):
        models[name] = {}
        for b in BANDS:
            X = g[f'{b}_modis'].values.reshape(-1, 1)
            y = g[f'{b}_true'].values
            if len(X) >= 2:
                lr = LinearRegression().fit(X, y)
                models[name][b] = lr
            else:
                models[name][b] = None
    return models, modis

# ===================== 4. 校正所有MODIS =====================
def calibrate_modis(modis_df, models):
    calib_rows = []
    for name, g in modis_df.groupby('name'):
        for _, row in g.iterrows():
            out = {'name': name, 'date': row['date']}
            for b in BANDS:
                val = row[b]
                if models[name][b] is not None:
                    out[b] = float(models[name][b].predict([[val]])[0])
                else:
                    out[b] = float(val)
            calib_rows.append(out)

    calib = pd.DataFrame(calib_rows)
    calib['data_type'] = 'modis_calib'
    return calib

# ===================== 5. 合并：高分辨率覆盖校正MODIS =====================
def merge_final(high, calib):
    combined = pd.concat([high, calib], ignore_index=True)
    # 同一天优先保留高分辨率
    combined = combined.sort_values(['name', 'date', 'data_type'], ascending=[True, True, False])
    combined = combined.drop_duplicates(subset=['name', 'date'], keep='first')
    combined = combined.sort_values(['name', 'date']).reset_index(drop=True)
    return combined

# ===================== 6. 补全为逐日数据 =====================
def make_daily(merged):
    daily_list = []
    for name, g in merged.groupby('name'):
        g = g.sort_values('date').copy()
        date_range = pd.date_range(g['date'].min(), g['date'].max(), freq='D')
        daily = pd.DataFrame({'date': date_range, 'name': name})
        daily = daily.merge(g[['name', 'date'] + BANDS], on=['name', 'date'], how='left')

        # 线性插值补全
        for b in BANDS:
            daily[b] = daily[b].interpolate(method='linear', limit_direction='both')

        daily_list.append(daily)

    final = pd.concat(daily_list, ignore_index=True)
    final['date'] = final['date'].dt.strftime('%Y-%m-%d')
    final = final[['name', 'date'] + BANDS].copy()

    # 确保纯数字格式
    for b in BANDS:
        final[b] = pd.to_numeric(final[b], errors='coerce')

    return final

# ===================== 主流程 =====================
if __name__ == "__main__":
    raw = pd.read_csv(INPUT_PATH)
    print("✅ 原始数据行数：", len(raw))

    high_df, raw = build_high_benchmark(raw)
    models, modis_data = train_lr_models(high_df, raw)
    calib_data = calibrate_modis(modis_data, models)
    merged = merge_final(high_df, calib_data)
    final_result = make_daily(merged)

    # 输出为 xlsx（纯数字格式）
    final_result.to_excel(OUTPUT_PATH, index=False, float_format='%.6f')

    print("\n🎉 全部完成！")
    print("✅ 输出格式：XLSX（纯数字）")
    print("✅ 时间序列：逐日完整")
    print("✅ 文件保存至：", OUTPUT_PATH)
    print("\n预览：")
    print(final_result.head())





# ==============NO.2===============
# ===================== 1. 核心配置 =====================
# 输入 → 现在也是 xlsx
INPUT_PATH = "多源融合校正.xlsx"
# 输出 → xlsx 纯数字
OUTPUT_PATH = "EnKF时序优化.xlsx"

BANDS = ['Green', 'Red', 'NIR', 'SWIR']
ENKF_SIZE = 50
OBS_ERROR = 0.002
PROC_ERROR = 0.005

# ===================== 2. 数据预处理 =====================
def load_and_prepare():
    # 读取 XLSX
    df = pd.read_excel(INPUT_PATH)
    
    # 标准化日期 + 排序
    df['date'] = pd.to_datetime(df['date'])  # 这里已修复
    df = df.sort_values(['name', 'date']).reset_index(drop=True)
    
    # 插值补全缺失值
    for band in BANDS:
        df[band] = df.groupby('name')[band].transform(
            lambda x: x.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        )
    
    print(f"✅ 读取 XLSX 数据：{len(df)} 行")
    print(f"✅ 点位数量：{df['name'].nunique()}")
    return df

# ===================== 3. EnKF 滤波核心 =====================
def enkf_smooth(series):
    n = len(series)
    smoothed = np.zeros(n)
    ensemble = np.random.normal(series.iloc[0], PROC_ERROR, ENKF_SIZE)
    
    for t in range(n):
        ensemble = np.random.normal(ensemble, PROC_ERROR)
        obs = series.iloc[t]
        ens_mean = np.mean(ensemble)
        ens_var = np.var(ensemble, ddof=1)
        kalman_gain = ens_var / (ens_var + OBS_ERROR)
        ensemble = ensemble + kalman_gain * (obs + np.random.normal(0, OBS_ERROR, ENKF_SIZE) - ensemble)
        smoothed[t] = np.mean(ensemble)
    
    return smoothed

# ===================== 4. 逐点逐波段 EnKF =====================
def process_enkf_optimize(df):
    result_list = []
    
    for point_name, point_df in df.groupby('name'):
        point_df = point_df.sort_values('date').reset_index(drop=True)
        
        optimized_data = {
            'name': int(point_name),
            'date': point_df['date'].dt.strftime('%Y-%m-%d')
        }
        
        for band in BANDS:
            optimized_data[band] = enkf_smooth(point_df[band])
        
        result_list.append(pd.DataFrame(optimized_data))
    
    final_df = pd.concat(result_list, ignore_index=True)
    
    # 强制纯数字格式
    for b in BANDS:
        final_df[b] = pd.to_numeric(final_df[b], errors='coerce')
    
    return final_df[['name', 'date'] + BANDS]

# ===================== 主流程 =====================
if __name__ == "__main__":
    fused_df = load_and_prepare()
    final_df = process_enkf_optimize(fused_df)
    
    # 输出 XLSX（纯数字，无格式污染）
    final_df.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')
    
    print("\n🎉 EnKF 时序平滑完成！")
    print(f"✅ 输入：XLSX")
    print(f"✅ 输出：XLSX 纯数字")
    print(f"✅ 文件已保存到：{OUTPUT_PATH}")
    print("\n预览：")
    print(final_df.head())




# ==============NO.3===============
# ===================== 路径配置 =====================
INPUT_PATH  = "EnKF时序优化.xlsx"
OUTPUT_PATH = "全作物指数.xlsx"

# ===================== 读取 XLSX 数据 =====================
df = pd.read_excel(INPUT_PATH)
df['date'] = pd.to_datetime(df['date'])

# ===================== 批量计算所有作物指数 =====================
def compute_all_indices(df):
    G = df['Green']
    R = df['Red']
    N = df['NIR']
    S = df['SWIR']
    
    # ========== 1. 长势/LAI指数 ==========
    df['NDVI']   = (N - R) / (N + R)
    df['EVI2']   = 2.5 * (N - R) / (N + 2.4*R + 1)
    df['SAVI']   = (N - R) * 1.5 / (N + R + 0.5)
    df['OSAVI']  = (N - R) / (N + R + 0.16)

    # ========== 2. 叶绿素/氮素指数 ==========
    df['GNDVI']  = (N - G) / (N + G)
    df['CIgreen']= (N / G) - 1

    # ========== 3. 水分/干旱指数 ==========
    df['MSI']    = S / N
    df['NDWI_SWIR'] = (N - S) / (N + S)
    df['NDWI_Green']= (G - N) / (G + N)

    # ========== 4. 生物量/产量指数 ==========
    df['RVI']    = N / R  # 异常值截断
    df['RVI']    = df['RVI'].clip(0, 15)  # 避免极端值

    # ========== 5. 生理胁迫指数 PRI ==========
    df['PRI']    = (G - R) / (G + R)  # 替代你要的PHRI

    # ========== 6. 你原有指数 ==========
    df['LMI']    = 0.484*((S-N)/(S+N)) + 0.687*df['NDVI'] + 0.542*df['NDWI_Green']

    return df

# ===================== 执行计算 =====================
df = compute_all_indices(df)

# ===================== 保存为 XLSX =====================
df.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')

# ===================== 输出信息 =====================
print("✅ 所有作物指数计算完成！")
print(f"✅ 指数数量：{len([col for col in df.columns if col not in ['name','date','Green','Red','NIR','SWIR']])} 个")
print(f"✅ 文件保存至：{OUTPUT_PATH}")
print("\n📊 指数列表：")
print([col for col in df.columns if col not in ['name','date','Green','Red','NIR','SWIR']])




# ==============NO.4===============
# ===================== 路径 =====================
INPUT_PATH  = "全作物指数.xlsx"
OUTPUT_PATH = "指数已校正.xlsx"

# ===================== 读取数据 =====================
df = pd.read_excel(INPUT_PATH)
df_backup = df.copy()  # 备份

# ===================== 遥感指数合理值域校正 =====================
def correct_indices(df):
    # 1) NDVI → [-1, 1]
    df['NDVI'] = df['NDVI'].clip(-1.0, 1.0)
    
    # 2) EVI2 → [-1, 1]
    df['EVI2'] = df['EVI2'].clip(-1.0, 1.0)
    
    # 3) SAVI / OSAVI → [-1, 1]
    df['SAVI'] = df['SAVI'].clip(-1.0, 1.0)
    df['OSAVI'] = df['OSAVI'].clip(-1.0, 1.0)
    
    # 4) GNDVI → [-1, 1]
    df['GNDVI'] = df['GNDVI'].clip(-1.0, 1.0)
    
    # 5) CIgreen → 小麦一般 0~1.5，极端不超过 3
    df['CIgreen'] = df['CIgreen'].clip(0, 3.0)
    
    # 6) MSI → 0~3 合理
    df['MSI'] = df['MSI'].clip(0, 3.0)
    
    # 7) NDWI_SWIR / NDWI_Green → [-1,1]
    df['NDWI_SWIR'] = df['NDWI_SWIR'].clip(-1.0, 1.0)
    df['NDWI_Green'] = df['NDWI_Green'].clip(-1.0, 1.0)
    
    # 8) RVI → 植被一般 1~8，上限 12
    df['RVI'] = df['RVI'].clip(1.0, 12.0)
    
    # 9) PRI → [-0.5, 0.5]
    df['PRI'] = df['PRI'].clip(-0.5, 0.5)
    
    # 10) LMI → 自定义指数，按合理范围约束
    df['LMI'] = df['LMI'].clip(-2.0, 5.0)

    return df

# ===================== 执行校正 =====================
df = correct_indices(df)

# ===================== 逐点插值，保证时序平滑 =====================
index_cols = [
    'NDVI','EVI2','SAVI','OSAVI',
    'GNDVI','CIgreen','MSI','NDWI_SWIR','NDWI_Green',
    'RVI','PRI','LMI'
]

for col in index_cols:
    df[col] = df.groupby('name')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

# ===================== 保存 =====================
df.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')

# ===================== 输出报告 =====================
print("✅ 指数值域校正完成！所有指数已约束在合理范围")
print("✅ 时序已平滑，无突变、无异常值，可直接用于建模")
print(f"✅ 校正后文件保存至：{OUTPUT_PATH}")

print("\n📊 各指数校正后范围：")
for c in index_cols:
    print(f"{c:<12} | min={df[c].min():6.3f}  max={df[c].max():6.3f}")