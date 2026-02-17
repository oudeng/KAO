# mimic_extract_v7.py
"""
MIMIC-IV ICU Mortality Risk Dataset Extractor

目标：提取与合成ICU死亡率数据集相同结构的真实数据
特征包括：MAP、乳酸、PF_ratio、GCS、肌酐、年龄、Charlson指数、入院时长、血管活性药物剂量、机械通气
目标变量：ICU死亡率（30天死亡率）

基于临床经验和需求的主要改进：
- 聚焦于ICU死亡率预测的关键临床特征
- 包含阈值效应相关的特征工程
- 计算实际的Charlson共病指数
- 提取血管活性药物的实际剂量
- 计算PaO2/FiO2比值
"""

import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MIMICMortalityExtractorV7:
    def __init__(self, db_config):
        """初始化数据库连接"""
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """建立数据库连接"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            self.cursor = self.conn.cursor()
            print("数据库连接成功!")
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def get_icu_stays_with_mortality(self, n_samples=5000):
        """
        获取ICU住院记录，包括死亡率信息
        选择有足够数据质量的患者
        """
        print("正在提取ICU住院记录...")
        
        query = """
        WITH icu_mortality AS (
            SELECT 
                ie.subject_id,
                ie.hadm_id,
                ie.stay_id,
                ie.intime,
                ie.outtime,
                ie.los,
                p.gender,
                p.anchor_age as age_years,
                adm.deathtime,
                adm.hospital_expire_flag,
                -- 计算ICU死亡和30天死亡
                CASE WHEN adm.deathtime BETWEEN ie.intime AND ie.outtime THEN 1 ELSE 0 END as icu_death,
                CASE WHEN adm.deathtime <= ie.intime + INTERVAL '30 days' THEN 1 ELSE 0 END as death_30days
            FROM mimiciv_icu.icustays ie
            JOIN mimiciv_hosp.patients p ON ie.subject_id = p.subject_id
            JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
            WHERE ie.los >= 1  -- 至少住院1天
            AND p.anchor_age >= 18  -- 成年患者
            AND p.anchor_age <= 95  -- 排除极高龄
        )
        SELECT *
        FROM icu_mortality
        ORDER BY RANDOM()
        LIMIT %s
        """
        
        self.cursor.execute(query, (n_samples,))
        results = self.cursor.fetchall()
        
        columns = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 
                  'los', 'gender', 'age_years', 'deathtime', 'hospital_expire_flag',
                  'icu_death', 'death_30days']
        
        df = pd.DataFrame(results, columns=columns)
        print(f"找到 {len(df)} 个ICU住院记录")
        print(f"ICU死亡率: {df['icu_death'].mean():.2%}")
        print(f"30天死亡率: {df['death_30days'].mean():.2%}")
        
        return df
    
    def extract_vital_signs(self, stay_ids):
        """
        提取生命体征数据（前24小时的统计值）
        包括MAP、心率、呼吸频率、SpO2
        """
        print("正在提取生命体征...")
        
        stay_ids_str = ','.join(map(str, stay_ids))
        
        query = f"""
        WITH first_24h AS (
            SELECT 
                ce.stay_id,
                ce.itemid,
                ce.valuenum,
                ce.charttime,
                ie.intime
            FROM mimiciv_icu.chartevents ce
            JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
            WHERE ce.stay_id IN ({stay_ids_str})
            AND ce.charttime <= ie.intime + INTERVAL '24 hours'
            AND ce.valuenum IS NOT NULL
            AND ce.itemid IN (
                220045,  -- Heart Rate
                220052, 220181,  -- MAP
                220179, 220050,  -- SBP
                220180, 220051,  -- DBP  
                220210, 224690,  -- Respiratory Rate
                220277  -- SpO2
            )
        )
        SELECT 
            stay_id,
            -- MAP (使用最低值，捕获低血压)
            MIN(CASE WHEN itemid IN (220052, 220181) THEN valuenum END) as MAP_min,
            AVG(CASE WHEN itemid IN (220052, 220181) THEN valuenum END) as MAP_mean,
            
            -- 心率
            MAX(CASE WHEN itemid = 220045 THEN valuenum END) as HR_max,
            AVG(CASE WHEN itemid = 220045 THEN valuenum END) as HR_mean,
            
            -- 收缩压和舒张压
            MIN(CASE WHEN itemid IN (220179, 220050) THEN valuenum END) as SBP_min,
            MIN(CASE WHEN itemid IN (220180, 220051) THEN valuenum END) as DBP_min,
            
            -- 呼吸频率
            MAX(CASE WHEN itemid IN (220210, 224690) THEN valuenum END) as RespRate_max,
            
            -- SpO2
            MIN(CASE WHEN itemid = 220277 THEN valuenum END) as SpO2_min
        FROM first_24h
        GROUP BY stay_id
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['stay_id', 'MAP_min', 'MAP_mean', 'HR_max', 'HR_mean', 
                  'SBP_min', 'DBP_min', 'RespRate_max', 'SpO2_min']
        
        df = pd.DataFrame(results, columns=columns)
        
        # 清理异常值
        df['MAP_min'] = df['MAP_min'].clip(30, 200)
        df['MAP_mean'] = df['MAP_mean'].clip(30, 200)
        df['HR_max'] = df['HR_max'].clip(30, 250)
        df['SBP_min'] = df['SBP_min'].clip(50, 250)
        df['DBP_min'] = df['DBP_min'].clip(20, 150)
        df['RespRate_max'] = df['RespRate_max'].clip(5, 60)
        df['SpO2_min'] = df['SpO2_min'].clip(50, 100)
        
        return df
    
    def extract_labs(self, stay_ids):
        """
        提取实验室检查结果（前24小时）
        重点关注乳酸和肌酐
        """
        print("正在提取实验室数据...")
        
        stay_ids_str = ','.join(map(str, stay_ids))
        
        query = f"""
        WITH first_24h_labs AS (
            SELECT 
                ie.stay_id,
                le.itemid,
                le.valuenum,
                le.charttime,
                ie.intime
            FROM mimiciv_hosp.labevents le
            JOIN mimiciv_icu.icustays ie ON le.subject_id = ie.subject_id
            WHERE ie.stay_id IN ({stay_ids_str})
            AND le.charttime BETWEEN ie.intime AND ie.intime + INTERVAL '24 hours'
            AND le.valuenum IS NOT NULL
            AND le.itemid IN (
                50813,  -- Lactate
                50912,  -- Creatinine
                50983, 50824,  -- Sodium
                51222, 51265,  -- Hemoglobin
                51301,  -- WBC
                51144   -- Bands
            )
        )
        SELECT 
            stay_id,
            -- 乳酸（使用最高值，捕获组织缺氧）
            MAX(CASE WHEN itemid = 50813 THEN valuenum END) as lactate_max,
            AVG(CASE WHEN itemid = 50813 THEN valuenum END) as lactate_mean,
            
            -- 肌酐（使用最高值，捕获肾功能损伤）
            MAX(CASE WHEN itemid = 50912 THEN valuenum END) as creatinine_max,
            
            -- 钠
            MIN(CASE WHEN itemid IN (50983, 50824) THEN valuenum END) as sodium_min,
            MAX(CASE WHEN itemid IN (50983, 50824) THEN valuenum END) as sodium_max,
            
            -- 血红蛋白
            MIN(CASE WHEN itemid IN (51222, 51265) THEN valuenum END) as hemoglobin_min,
            
            -- 白细胞
            MAX(CASE WHEN itemid = 51301 THEN valuenum END) as wbc_max,
            
            -- 中性粒细胞
            MAX(CASE WHEN itemid = 51144 THEN valuenum END) as bands_max
        FROM first_24h_labs
        GROUP BY stay_id
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['stay_id', 'lactate_max', 'lactate_mean', 'creatinine_max',
                  'sodium_min', 'sodium_max', 'hemoglobin_min', 'wbc_max', 'bands_max']
        
        df = pd.DataFrame(results, columns=columns)
        
        # 清理异常值
        df['lactate_max'] = df['lactate_max'].clip(0.1, 30)
        df['lactate_mean'] = df['lactate_mean'].clip(0.1, 30)
        df['creatinine_max'] = df['creatinine_max'].clip(0.1, 20)
        df['sodium_min'] = df['sodium_min'].clip(100, 180)
        df['hemoglobin_min'] = df['hemoglobin_min'].clip(3, 20)
        df['wbc_max'] = df['wbc_max'].clip(0, 100)
        
        return df
    
    def extract_gcs(self, stay_ids):
        """
        提取GCS评分（前24小时最低值）
        """
        print("正在提取GCS评分...")
        
        stay_ids_str = ','.join(map(str, stay_ids))
        
        query = f"""
        WITH gcs_components AS (
            SELECT 
                ce.stay_id,
                ce.charttime,
                MAX(CASE WHEN ce.itemid = 223900 THEN ce.valuenum END) as gcs_verbal,
                MAX(CASE WHEN ce.itemid = 223901 THEN ce.valuenum END) as gcs_motor,
                MAX(CASE WHEN ce.itemid = 227013 THEN ce.valuenum END) as gcs_eye,
                MAX(CASE WHEN ce.itemid = 220739 THEN ce.valuenum END) as gcs_total_direct
            FROM mimiciv_icu.chartevents ce
            JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
            WHERE ce.stay_id IN ({stay_ids_str})
            AND ce.charttime <= ie.intime + INTERVAL '24 hours'
            AND ce.itemid IN (223900, 223901, 227013, 220739)
            GROUP BY ce.stay_id, ce.charttime
        ),
        gcs_calculated AS (
            SELECT 
                stay_id,
                charttime,
                COALESCE(gcs_total_direct, 
                        gcs_verbal + gcs_motor + gcs_eye) as gcs_total
            FROM gcs_components
        )
        SELECT 
            stay_id,
            MIN(gcs_total) as GCS_min,
            AVG(gcs_total) as GCS_mean
        FROM gcs_calculated
        WHERE gcs_total IS NOT NULL
        GROUP BY stay_id
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['stay_id', 'GCS_min', 'GCS_mean']
        df = pd.DataFrame(results, columns=columns)
        
        # 确保GCS在合理范围内
        df['GCS_min'] = df['GCS_min'].clip(3, 15)
        df['GCS_mean'] = df['GCS_mean'].clip(3, 15)
        
        return df
    
    def extract_pao2_fio2(self, stay_ids):
        """
        计算PaO2/FiO2比值（氧合指数）
        """
        print("正在计算PaO2/FiO2比值...")
        
        stay_ids_str = ','.join(map(str, stay_ids))
        
        query = f"""
        WITH blood_gas AS (
            SELECT 
                ce.stay_id,
                ce.charttime,
                MAX(CASE WHEN ce.itemid = 220224 THEN ce.valuenum END) as pao2,
                MAX(CASE WHEN ce.itemid IN (223835, 220277) THEN ce.valuenum END) as fio2_chart
            FROM mimiciv_icu.chartevents ce
            JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
            WHERE ce.stay_id IN ({stay_ids_str})
            AND ce.charttime <= ie.intime + INTERVAL '24 hours'
            AND ce.itemid IN (220224, 223835, 220277)
            GROUP BY ce.stay_id, ce.charttime
        ),
        pf_ratio AS (
            SELECT 
                stay_id,
                pao2 / GREATEST(fio2_chart/100, 0.21) as pf_ratio
            FROM blood_gas
            WHERE pao2 IS NOT NULL 
            AND fio2_chart IS NOT NULL
        )
        SELECT 
            stay_id,
            MIN(pf_ratio) as PF_ratio_min,
            AVG(pf_ratio) as PF_ratio_mean
        FROM pf_ratio
        WHERE pf_ratio > 0 AND pf_ratio < 800
        GROUP BY stay_id
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['stay_id', 'PF_ratio_min', 'PF_ratio_mean']
        df = pd.DataFrame(results, columns=columns)
        
        return df
    
    def extract_vasopressors(self, stay_ids):
        """
        提取血管活性药物使用情况和剂量
        """
        print("正在提取血管活性药物数据...")
        
        stay_ids_str = ','.join(map(str, stay_ids))
        
        query = f"""
        WITH vasopressor_doses AS (
            SELECT 
                inp.stay_id,
                inp.itemid,
                inp.rate,
                inp.starttime,
                inp.endtime,
                ie.intime,
                -- 转换为去甲肾上腺素当量 (mcg/kg/min)
                CASE 
                    WHEN inp.itemid = 221906 THEN inp.rate  -- Norepinephrine
                    WHEN inp.itemid = 221662 THEN inp.rate * 0.01  -- Dopamine
                    WHEN inp.itemid = 221289 THEN inp.rate  -- Epinephrine
                    WHEN inp.itemid = 221749 THEN inp.rate * 2.5  -- Phenylephrine
                    WHEN inp.itemid = 222315 THEN inp.rate * 100  -- Vasopressin
                    ELSE inp.rate
                END as norepinephrine_equivalent
            FROM mimiciv_icu.inputevents inp
            JOIN mimiciv_icu.icustays ie ON inp.stay_id = ie.stay_id
            WHERE inp.stay_id IN ({stay_ids_str})
            AND inp.starttime <= ie.intime + INTERVAL '24 hours'
            AND inp.itemid IN (
                221906,  -- Norepinephrine
                221662,  -- Dopamine
                221289,  -- Epinephrine
                221749,  -- Phenylephrine
                222315   -- Vasopressin
            )
            AND inp.rate > 0
        )
        SELECT 
            stay_id,
            COUNT(DISTINCT itemid) as vasopressor_types,
            MAX(norepinephrine_equivalent) as vasopressor_dose_max,
            AVG(norepinephrine_equivalent) as vasopressor_dose_mean,
            SUM(EXTRACT(EPOCH FROM (endtime - starttime))/3600) as vasopressor_hours
        FROM vasopressor_doses
        GROUP BY stay_id
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['stay_id', 'vasopressor_types', 'vasopressor_dose_max', 
                  'vasopressor_dose_mean', 'vasopressor_hours']
        df = pd.DataFrame(results, columns=columns)
        
        # 创建二元指标
        df['vasopressor_use'] = 1
        
        return df
    
    def extract_mechanical_ventilation(self, stay_ids):
        """
        提取机械通气使用情况
        """
        print("正在提取机械通气数据...")
        
        stay_ids_str = ','.join(map(str, stay_ids))
        
        query = f"""
        WITH vent_settings AS (
            SELECT DISTINCT
                ce.stay_id,
                1 as mechanical_ventilation
            FROM mimiciv_icu.chartevents ce
            JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
            WHERE ce.stay_id IN ({stay_ids_str})
            AND ce.charttime <= ie.intime + INTERVAL '24 hours'
            AND ce.itemid IN (
                224684,  -- Tidal Volume (Set)
                224685,  -- Tidal Volume (Observed)
                220339,  -- PEEP set
                224700,  -- PEEP
                223835,  -- FiO2
                223849   -- Vent Mode
            )
            AND ce.valuenum IS NOT NULL
        )
        SELECT * FROM vent_settings
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['stay_id', 'mechanical_ventilation']
        df = pd.DataFrame(results, columns=columns)
        
        return df
    
    def calculate_charlson_index(self, subject_ids):
        """
        计算Charlson共病指数
        基于ICD诊断代码
        """
        print("正在计算Charlson共病指数...")
        
        subject_ids_str = ','.join(map(str, subject_ids))
        
        query = f"""
        WITH charlson_diagnoses AS (
            SELECT 
                d.subject_id,
                d.icd_code,
                d.icd_version,
                CASE 
                    -- Myocardial infarction
                    WHEN (d.icd_version = 9 AND d.icd_code LIKE '410%') OR
                         (d.icd_version = 10 AND d.icd_code LIKE 'I21%') THEN 1
                    -- Congestive heart failure
                    WHEN (d.icd_version = 9 AND d.icd_code IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4254','4255','4256','4257','4258','4259','428')) OR
                         (d.icd_version = 10 AND d.icd_code LIKE 'I50%') THEN 1
                    -- Peripheral vascular disease
                    WHEN (d.icd_version = 9 AND d.icd_code IN ('0930','4373','440','441','4431','4432','4438','4439','4471','5571','5579','V434')) OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'I70%' OR d.icd_code LIKE 'I71%')) THEN 1
                    -- Cerebrovascular disease
                    WHEN (d.icd_version = 9 AND d.icd_code BETWEEN '430' AND '438') OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'I60%' OR d.icd_code LIKE 'I61%' OR d.icd_code LIKE 'I62%' OR d.icd_code LIKE 'I63%')) THEN 1
                    -- Dementia
                    WHEN (d.icd_version = 9 AND d.icd_code IN ('290','2941','3312')) OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'F00%' OR d.icd_code LIKE 'F01%' OR d.icd_code LIKE 'F02%' OR d.icd_code LIKE 'F03%')) THEN 1
                    -- Chronic pulmonary disease
                    WHEN (d.icd_version = 9 AND d.icd_code BETWEEN '490' AND '505') OR
                         (d.icd_version = 10 AND (d.icd_code BETWEEN 'J40' AND 'J47' OR d.icd_code BETWEEN 'J60' AND 'J67')) THEN 1
                    -- Diabetes without complications
                    WHEN (d.icd_version = 9 AND d.icd_code LIKE '250%' AND SUBSTRING(d.icd_code, 5, 1) IN ('0','1','2','3','8','9')) OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%')) THEN 1
                    -- Diabetes with complications
                    WHEN (d.icd_version = 9 AND d.icd_code LIKE '250%' AND SUBSTRING(d.icd_code, 5, 1) IN ('4','5','6','7')) OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%')) THEN 2
                    -- Renal disease
                    WHEN (d.icd_version = 9 AND d.icd_code IN ('582','5830','5831','5832','5834','5836','5837','585','586','5880','V420','V451','V56')) OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'N18%' OR d.icd_code LIKE 'N19%')) THEN 2
                    -- Mild liver disease
                    WHEN (d.icd_version = 9 AND d.icd_code IN ('5712','5714','5715','5716','5718','5719')) OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'K70%' OR d.icd_code LIKE 'K71%')) THEN 1
                    -- Moderate/severe liver disease
                    WHEN (d.icd_version = 9 AND d.icd_code IN ('4560','4561','4562','5722','5723','5724','5728')) OR
                         (d.icd_version = 10 AND (d.icd_code LIKE 'K72%' OR d.icd_code LIKE 'K76%')) THEN 3
                    -- Cancer
                    WHEN (d.icd_version = 9 AND d.icd_code BETWEEN '140' AND '172' OR d.icd_code BETWEEN '174' AND '195') OR
                         (d.icd_version = 10 AND d.icd_code BETWEEN 'C00' AND 'C75') THEN 2
                    -- Metastatic cancer
                    WHEN (d.icd_version = 9 AND d.icd_code BETWEEN '196' AND '199') OR
                         (d.icd_version = 10 AND d.icd_code BETWEEN 'C77' AND 'C80') THEN 6
                    -- AIDS
                    WHEN (d.icd_version = 9 AND d.icd_code BETWEEN '042' AND '044') OR
                         (d.icd_version = 10 AND d.icd_code BETWEEN 'B20' AND 'B24') THEN 6
                    ELSE 0
                END as score
            FROM mimiciv_hosp.diagnoses_icd d
            WHERE d.subject_id IN ({subject_ids_str})
        )
        SELECT 
            subject_id,
            SUM(score) as charlson_index
        FROM charlson_diagnoses
        GROUP BY subject_id
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['subject_id', 'charlson_index']
        df = pd.DataFrame(results, columns=columns)
        
        return df
    
    def extract_urine_output(self, stay_ids):
        """
        提取尿量数据（前24小时）
        """
        print("正在提取尿量数据...")
        
        stay_ids_str = ','.join(map(str, stay_ids))
        
        query = f"""
        WITH hourly_urine AS (
            SELECT 
                oe.stay_id,
                DATE_TRUNC('hour', oe.charttime) as hour,
                SUM(oe.value) as hourly_output
            FROM mimiciv_icu.outputevents oe
            JOIN mimiciv_icu.icustays ie ON oe.stay_id = ie.stay_id
            WHERE oe.stay_id IN ({stay_ids_str})
            AND oe.charttime <= ie.intime + INTERVAL '24 hours'
            AND oe.value > 0
            AND oe.value < 5000  -- 排除异常值
            GROUP BY oe.stay_id, DATE_TRUNC('hour', oe.charttime)
        )
        SELECT 
            stay_id,
            MIN(hourly_output) as urine_output_min,
            AVG(hourly_output) as urine_output_mean,
            SUM(hourly_output) as urine_output_total_24h
        FROM hourly_urine
        GROUP BY stay_id
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        columns = ['stay_id', 'urine_output_min', 'urine_output_mean', 'urine_output_total_24h']
        df = pd.DataFrame(results, columns=columns)
        
        return df
    
    def combine_features(self, icu_stays):
        """
        合并所有特征，创建最终数据集
        """
        print("\n正在合并所有特征...")
        
        # 获取需要的ID列表
        stay_ids = icu_stays['stay_id'].tolist()
        subject_ids = icu_stays['subject_id'].unique().tolist()
        
        # 提取各类特征
        vital_signs = self.extract_vital_signs(stay_ids)
        labs = self.extract_labs(stay_ids)
        gcs = self.extract_gcs(stay_ids)
        pf_ratio = self.extract_pao2_fio2(stay_ids)
        vasopressors = self.extract_vasopressors(stay_ids)
        ventilation = self.extract_mechanical_ventilation(stay_ids)
        charlson = self.calculate_charlson_index(subject_ids)
        urine = self.extract_urine_output(stay_ids)
        
        # 合并所有数据
        final_df = icu_stays.copy()
        
        # 合并特征
        for df in [vital_signs, labs, gcs, pf_ratio, vasopressors, ventilation, urine]:
            if not df.empty:
                final_df = final_df.merge(df, on='stay_id', how='left')
        
        # 合并Charlson指数
        if not charlson.empty:
            final_df = final_df.merge(charlson, on='subject_id', how='left')
        
        # 计算入院时长（小时）
        final_df['hours_since_admission'] = 24  # 使用前24小时数据
        
        # 填充缺失值
        final_df['vasopressor_use'] = final_df['vasopressor_use'].fillna(0)
        final_df['vasopressor_dose_max'] = final_df['vasopressor_dose_max'].fillna(0)
        final_df['mechanical_ventilation'] = final_df['mechanical_ventilation'].fillna(0)
        final_df['charlson_index'] = final_df['charlson_index'].fillna(0)
        
        # 重命名列以匹配合成数据集
        rename_dict = {
            'MAP_min': 'MAP_mmHg',
            'lactate_max': 'lactate_mmol_L',
            'PF_ratio_min': 'PF_ratio',
            'GCS_min': 'GCS',
            'creatinine_max': 'creatinine_mg_dL',
            'vasopressor_dose_max': 'vasopressor_dose',
            'death_30days': 'mortality_risk'  # 使用30天死亡率作为目标
        }
        
        final_df = final_df.rename(columns=rename_dict)
        
        # 选择最终特征（与合成数据集保持一致）
        feature_columns = [
            'MAP_mmHg', 'lactate_mmol_L', 'PF_ratio', 'GCS', 'creatinine_mg_dL',
            'age_years', 'charlson_index', 'hours_since_admission',
            'vasopressor_dose', 'mechanical_ventilation', 'mortality_risk'
        ]
        
        # 添加额外的有用特征
        additional_columns = [
            'stay_id', 'subject_id', 'gender', 'SBP_min', 'DBP_min', 'HR_max',
            'RespRate_max', 'SpO2_min', 'lactate_mean', 'sodium_min', 'hemoglobin_min',
            'wbc_max', 'urine_output_min', 'vasopressor_use', 'icu_death'
        ]
        
        # 选择存在的列
        available_columns = [col for col in feature_columns + additional_columns if col in final_df.columns]
        final_df = final_df[available_columns]
        
        # 删除缺失关键特征的行
        critical_features = ['MAP_mmHg', 'lactate_mmol_L', 'GCS', 'creatinine_mg_dL']
        for feature in critical_features:
            if feature in final_df.columns:
                initial_count = len(final_df)
                final_df = final_df.dropna(subset=[feature])
                dropped = initial_count - len(final_df)
                if dropped > 0:
                    print(f"删除 {dropped} 行由于缺失 {feature}")
        
        return final_df
    
    def create_threshold_features(self, df):
        """
        创建阈值相关的特征，增强阈值效应的可检测性
        """
        print("创建阈值特征...")
        
        # MAP < 65 mmHg (低血压)
        if 'MAP_mmHg' in df.columns:
            df['MAP_below_65'] = (df['MAP_mmHg'] < 65).astype(int)
            df['MAP_severely_low'] = (df['MAP_mmHg'] < 50).astype(int)
        
        # 乳酸 > 2 mmol/L (组织缺氧)
        if 'lactate_mmol_L' in df.columns:
            df['lactate_elevated'] = (df['lactate_mmol_L'] > 2).astype(int)
            df['lactate_severely_elevated'] = (df['lactate_mmol_L'] > 4).astype(int)
        
        # PF_ratio < 300 (ARDS)
        if 'PF_ratio' in df.columns:
            df['ARDS_mild'] = (df['PF_ratio'] < 300).astype(int)
            df['ARDS_moderate'] = (df['PF_ratio'] < 200).astype(int)
            df['ARDS_severe'] = (df['PF_ratio'] < 100).astype(int)
        
        # GCS < 8 (重度昏迷)
        if 'GCS' in df.columns:
            df['GCS_severe'] = (df['GCS'] < 8).astype(int)
            df['GCS_moderate'] = ((df['GCS'] >= 8) & (df['GCS'] < 13)).astype(int)
        
        # 肌酐 > 1.5 mg/dL (急性肾损伤)
        if 'creatinine_mg_dL' in df.columns:
            df['AKI_stage1'] = (df['creatinine_mg_dL'] > 1.5).astype(int)
            df['AKI_stage2'] = (df['creatinine_mg_dL'] > 2.0).astype(int)
            df['AKI_stage3'] = (df['creatinine_mg_dL'] > 3.0).astype(int)
        
        # 年龄 > 65 (老年)
        if 'age_years' in df.columns:
            df['elderly'] = (df['age_years'] > 65).astype(int)
            df['very_elderly'] = (df['age_years'] > 80).astype(int)
        
        # 休克指数 (HR/SBP)
        if 'HR_max' in df.columns and 'SBP_min' in df.columns:
            df['shock_index'] = df['HR_max'] / df['SBP_min'].replace(0, np.nan)
            df['shock_index_high'] = (df['shock_index'] > 0.9).astype(int)
        
        # 复合风险评分（类似SOFA）
        risk_score = 0
        if 'MAP_below_65' in df.columns:
            risk_score += df['MAP_below_65']
        if 'lactate_elevated' in df.columns:
            risk_score += df['lactate_elevated'] * 2  # 加权
        if 'ARDS_mild' in df.columns:
            risk_score += df['ARDS_mild']
        if 'GCS_severe' in df.columns:
            risk_score += df['GCS_severe'] * 2
        if 'AKI_stage1' in df.columns:
            risk_score += df['AKI_stage1']
        if 'vasopressor_use' in df.columns:
            risk_score += df['vasopressor_use'] * 2
        
        df['composite_risk_score'] = risk_score
        
        return df
    
    def analyze_dataset_quality(self, df):
        """
        分析数据集质量和特征分布
        """
        print("\n=== 数据集质量分析 ===")
        print(f"总样本数: {len(df)}")
        print(f"特征数: {len(df.columns)}")
        
        # 目标变量分布
        if 'mortality_risk' in df.columns:
            print(f"\n死亡率分布:")
            print(f"  30天死亡率: {df['mortality_risk'].mean():.2%}")
            if 'icu_death' in df.columns:
                print(f"  ICU死亡率: {df['icu_death'].mean():.2%}")
        
        # 关键特征的缺失率
        print(f"\n特征完整性:")
        critical_features = ['MAP_mmHg', 'lactate_mmol_L', 'PF_ratio', 'GCS', 'creatinine_mg_dL']
        for feature in critical_features:
            if feature in df.columns:
                missing_rate = df[feature].isna().mean()
                print(f"  {feature}: {(1-missing_rate):.1%} 完整")
        
        # 阈值分布
        print(f"\n临床阈值分布:")
        if 'MAP_below_65' in df.columns:
            print(f"  MAP < 65 mmHg: {df['MAP_below_65'].mean():.1%}")
        if 'lactate_elevated' in df.columns:
            print(f"  乳酸 > 2 mmol/L: {df['lactate_elevated'].mean():.1%}")
        if 'ARDS_mild' in df.columns:
            print(f"  PF_ratio < 300: {df['ARDS_mild'].mean():.1%}")
        if 'GCS_severe' in df.columns:
            print(f"  GCS < 8: {df['GCS_severe'].mean():.1%}")
        if 'AKI_stage1' in df.columns:
            print(f"  肌酐 > 1.5 mg/dL: {df['AKI_stage1'].mean():.1%}")
        
        # 基本统计
        print(f"\n关键特征统计:")
        stats_features = ['MAP_mmHg', 'lactate_mmol_L', 'PF_ratio', 'GCS', 'creatinine_mg_dL', 'age_years']
        for feature in stats_features:
            if feature in df.columns:
                print(f"  {feature}:")
                print(f"    均值: {df[feature].mean():.2f}")
                print(f"    中位数: {df[feature].median():.2f}")
                print(f"    标准差: {df[feature].std():.2f}")
                print(f"    范围: [{df[feature].min():.2f}, {df[feature].max():.2f}]")
        
        return df
    
    def export_dataset(self, df, filename='mimic_icu_mortality_dataset.csv'):
        """
        导出最终数据集
        """
        print(f"\n导出数据集到 {filename}")
        
        # 确保列顺序与合成数据集一致
        primary_columns = [
            'MAP_mmHg', 'lactate_mmol_L', 'PF_ratio', 'GCS', 'creatinine_mg_dL',
            'age_years', 'charlson_index', 'hours_since_admission',
            'vasopressor_dose', 'mechanical_ventilation', 'mortality_risk'
        ]
        
        # 添加阈值特征
        threshold_columns = [
            'MAP_below_65', 'lactate_elevated', 'lactate_severely_elevated',
            'ARDS_mild', 'ARDS_moderate', 'GCS_severe', 'AKI_stage1',
            'composite_risk_score'
        ]
        
        # 添加其他有用特征
        additional_columns = [
            'stay_id', 'subject_id', 'gender', 'SBP_min', 'DBP_min', 'HR_max',
            'RespRate_max', 'SpO2_min', 'hemoglobin_min', 'sodium_min',
            'urine_output_min', 'vasopressor_use', 'icu_death'
        ]
        
        # 选择存在的列
        export_columns = []
        for col_list in [primary_columns, threshold_columns, additional_columns]:
            for col in col_list:
                if col in df.columns and col not in export_columns:
                    export_columns.append(col)
        
        export_df = df[export_columns]
        
        # 导出CSV
        export_df.to_csv(filename, index=False)
        print(f"数据集已保存: {filename}")
        print(f"包含 {len(export_df)} 个样本, {len(export_columns)} 个特征")
        
        # 同时导出数据字典
        dict_filename = filename.replace('.csv', '_dictionary.txt')
        with open(dict_filename, 'w', encoding='utf-8') as f:
            f.write("MIMIC-IV ICU死亡率预测数据集 - 数据字典\n")
            f.write("="*50 + "\n\n")
            
            f.write("主要特征（与合成数据集对应）:\n")
            f.write("-"*30 + "\n")
            f.write("MAP_mmHg: 平均动脉压最低值 (mmHg)\n")
            f.write("lactate_mmol_L: 乳酸最高值 (mmol/L)\n")
            f.write("PF_ratio: PaO2/FiO2比值最低值\n")
            f.write("GCS: 格拉斯哥昏迷评分最低值 (3-15)\n")
            f.write("creatinine_mg_dL: 肌酐最高值 (mg/dL)\n")
            f.write("age_years: 年龄 (岁)\n")
            f.write("charlson_index: Charlson共病指数\n")
            f.write("hours_since_admission: 入院后时间 (小时)\n")
            f.write("vasopressor_dose: 血管活性药物最大剂量 (去甲肾上腺素当量)\n")
            f.write("mechanical_ventilation: 机械通气 (0/1)\n")
            f.write("mortality_risk: 30天死亡率 (0/1)\n\n")
            
            f.write("阈值特征:\n")
            f.write("-"*30 + "\n")
            f.write("MAP_below_65: MAP < 65 mmHg (低血压)\n")
            f.write("lactate_elevated: 乳酸 > 2 mmol/L\n")
            f.write("lactate_severely_elevated: 乳酸 > 4 mmol/L\n")
            f.write("ARDS_mild: PF_ratio < 300 (轻度ARDS)\n")
            f.write("ARDS_moderate: PF_ratio < 200 (中度ARDS)\n")
            f.write("GCS_severe: GCS < 8 (重度昏迷)\n")
            f.write("AKI_stage1: 肌酐 > 1.5 mg/dL (急性肾损伤)\n")
            f.write("composite_risk_score: 复合风险评分\n")
        
        print(f"数据字典已保存: {dict_filename}")
        
        return export_df


def main():
    """主函数"""
    # 数据库配置
    db_config = {
        'host': 'localhost',
        'database': 'mimiciv',
        'user': 'postgres',
        'password': '1233214'
    }
    
    # 创建提取器
    extractor = MIMICMortalityExtractorV7(db_config)
    
    try:
        # 连接数据库
        if not extractor.connect():
            return
        
        print("="*60)
        print("MIMIC-IV ICU死亡率预测数据集提取")
        print("Version 7 - 对标合成数据集")
        print("="*60)
        
        # 1. 获取ICU住院记录
        icu_stays = extractor.get_icu_stays_with_mortality(n_samples=5000)
        
        if icu_stays.empty:
            print("未找到符合条件的ICU住院记录")
            return
        
        # 2. 提取和合并所有特征
        final_dataset = extractor.combine_features(icu_stays)
        
        if final_dataset.empty:
            print("特征提取失败")
            return
        
        print(f"\n成功提取 {len(final_dataset)} 个样本")
        
        # 3. 创建阈值特征
        final_dataset = extractor.create_threshold_features(final_dataset)
        
        # 4. 分析数据集质量
        final_dataset = extractor.analyze_dataset_quality(final_dataset)
        
        # 5. 导出数据集
        extractor.export_dataset(final_dataset, 'mimic_icu_mortality_real.csv')
        
        # 6. 创建与合成数据集相同格式的版本
        synthetic_format = final_dataset[[
            'MAP_mmHg', 'lactate_mmol_L', 'PF_ratio', 'GCS', 'creatinine_mg_dL',
            'age_years', 'charlson_index', 'hours_since_admission',
            'vasopressor_dose', 'mechanical_ventilation', 'mortality_risk'
        ]].copy()
        
        synthetic_format.to_csv('mimic_icu_mortality_synthetic_format.csv', index=False)
        print(f"\n已创建与合成数据集相同格式的版本: mimic_icu_mortality_synthetic_format.csv")
        
        print("\n"+"="*60)
        print("数据提取完成!")
        print("="*60)
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 关闭数据库连接
        extractor.close()


if __name__ == "__main__":
    main()