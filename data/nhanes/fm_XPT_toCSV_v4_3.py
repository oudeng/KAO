#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHANES XPT数据处理 v4.3
从XPT文件直接生成数据集（基于临床经验或需求选择适合阈值/门控机制的特征）
改进：严格空腹选项、鲁棒编码、winsorize、zscore元数据
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

warnings.filterwarnings('ignore')


class NHANESProcessor:
    def __init__(self, data_dir='nhanes_data'):
        self.data_dir = data_dir
        self.data = {}
        self.merged_df = None
        
    def load_xpt_files(self):
        """加载所有XPT文件"""
        print("="*60)
        print("加载NHANES XPT文件")
        print("="*60)
        
        # 定义需要的文件和变量
        files_to_load = {
            'DEMO_J.XPT': ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR'],
            'BMX_J.XPT': ['SEQN', 'BMXBMI', 'BMXWAIST', 'BMXWT', 'BMXHT'],
            'BPX_J.XPT': ['SEQN', 'BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2', 'BPXSY3', 'BPXDI3'],
            'HDL_J.XPT': ['SEQN', 'LBDHDD', 'LBDHDDSI'],
            'TRIGLY_J.XPT': ['SEQN', 'LBXTR', 'LBDTRSI'],
            'GLU_J.XPT': ['SEQN', 'LBXGLU', 'LBDGLUSI'],
            'GHB_J.XPT': ['SEQN', 'LBXGH'],
            'TCHOL_J.XPT': ['SEQN', 'LBXTC', 'LBDTCSI'],
            # v4新增文件
            'BPQ_J.XPT': ['SEQN', 'BPQ050A', 'BPQ090D'],  # 降压药、降脂药
            'DIQ_J.XPT': ['SEQN', 'DIQ070'],  # 降糖药
            'SMQ_J.XPT': ['SEQN', 'SMQ040'],  # 吸烟状态
            'FASTQX_J.XPT': ['SEQN', 'PHAFSTHR', 'PHAFSTMN']  # 空腹时间
        }
        
        for filename, desired_vars in files_to_load.items():
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                print(f"\n读取 {filename}...")
                try:
                    df, meta = pyreadstat.read_xport(filepath)
                    available_vars = [v for v in desired_vars if v in df.columns]
                    if not available_vars:
                        print(f"  ⚠️ 没有找到需要的变量")
                        continue
                    
                    df = df[available_vars]
                    key = filename.replace('.XPT', '')
                    self.data[key] = df
                    
                    print(f"  ✔ 成功: {len(df)} 行, {len(available_vars)} 个变量")
                    print(f"    变量: {', '.join(available_vars)}")
                    
                except Exception as e:
                    print(f"  ✗ 读取失败: {e}")
            else:
                print(f"  ⚠️ 文件不存在: {filename}")
        
        print(f"\n成功加载 {len(self.data)} 个数据集")
        return self.data
    
    def merge_datasets(self):
        """合并所有数据集"""
        if not self.data:
            print("错误：没有数据可以合并")
            return None
        
        print("\n" + "="*60)
        print("合并数据集")
        print("="*60)
        
        # 从DEMO开始
        if 'DEMO_J' in self.data:
            merged = self.data['DEMO_J'].copy()
            print(f"基础数据集 (DEMO_J): {len(merged)} 条记录")
            
            for name, df in self.data.items():
                if name != 'DEMO_J':
                    print(f"  合并 {name}...")
                    merged = merged.merge(df, on='SEQN', how='left')
        else:
            first_key = list(self.data.keys())[0]
            merged = self.data[first_key].copy()
            print(f"基础数据集 ({first_key}): {len(merged)} 条记录")
            
            for name, df in self.data.items():
                if name != first_key:
                    merged = merged.merge(df, on='SEQN', how='outer')
        
        self.merged_df = merged
        print(f"\n✔ 合并完成: {len(merged)} 行 × {len(merged.columns)} 列")
        return merged
    
    def standardize_and_process(self, strict_fasting=False, winsorize=False):
        """标准化单位并处理所有特征"""
        if self.merged_df is None:
            return None
        
        print("\n" + "="*60)
        print("处理数据")
        print("="*60)
        
        df = self.merged_df.copy()
        
        # 筛选成年人
        if 'RIDAGEYR' in df.columns:
            initial_count = len(df)
            df = df[df['RIDAGEYR'] >= 18]
            print(f"筛选成年人(≥18岁): {initial_count} → {len(df)}")
        
        # 处理血压（多次测量平均值）
        if 'BPXSY1' in df.columns:
            bp_sys_cols = [col for col in ['BPXSY1', 'BPXSY2', 'BPXSY3'] if col in df.columns]
            df['systolic_bp'] = df[bp_sys_cols].mean(axis=1)
            print(f"收缩压: 使用{len(bp_sys_cols)}次测量的平均值")
            
        if 'BPXDI1' in df.columns:
            bp_dia_cols = [col for col in ['BPXDI1', 'BPXDI2', 'BPXDI3'] if col in df.columns]
            df['diastolic_bp'] = df[bp_dia_cols].mean(axis=1)
            print(f"舒张压: 使用{len(bp_dia_cols)}次测量的平均值")
        
        # 单位转换
        if 'LBDTRSI' in df.columns and 'LBXTR' not in df.columns:
            df['LBXTR'] = df['LBDTRSI'] * 88.57
            print("甘油三酯: mmol/L → mg/dL")
        
        if 'LBDHDDSI' in df.columns and 'LBDHDD' not in df.columns:
            df['LBDHDD'] = df['LBDHDDSI'] * 38.67
            print("HDL胆固醇: mmol/L → mg/dL")
        
        if 'LBDTCSI' in df.columns and 'LBXTC' not in df.columns:
            df['LBXTC'] = df['LBDTCSI'] * 38.67
            print("总胆固醇: mmol/L → mg/dL")
        
        if 'LBDGLUSI' in df.columns and 'LBXGLU' not in df.columns:
            df['LBXGLU'] = df['LBDGLUSI'] * 18.02
            print("血糖: mmol/L → mg/dL")
        
        # 处理性别 (标准化到 0=男, 1=女)
        if 'RIAGENDR' in df.columns:
            gender = df['RIAGENDR'].copy()
            if gender.min() == 1 and gender.max() == 2:
                gender = gender - 1  # 1,2 → 0,1
            df['gender_std'] = gender.astype(float)
            print(f"性别编码: 0=男性, 1=女性")
        
        # 创建年龄分段
        if 'RIDAGEYR' in df.columns:
            bins = [18, 30, 40, 50, 60, 70, np.inf]
            labels = [0, 1, 2, 3, 4, 5]
            df['age_band'] = pd.cut(df['RIDAGEYR'], bins=bins, right=False, labels=labels).astype(float)
            print("年龄分段: 0=18-29, 1=30-39, 2=40-49, 3=50-59, 4=60-69, 5=70+")
        
        # 处理药物使用 (1=Yes, 2=No → 1,0)
        if 'BPQ050A' in df.columns:
            df['bp_med_std'] = (df['BPQ050A'] == 1).astype(float)
            
        if 'BPQ090D' in df.columns:
            df['lipid_med_std'] = (df['BPQ090D'] == 1).astype(float)
            
        if 'DIQ070' in df.columns:
            df['glucose_med_std'] = (df['DIQ070'] == 1).astype(float)
        
        # 处理吸烟 (1=Every day, 2=Some days → 1; 3=Not at all → 0)
        if 'SMQ040' in df.columns:
            df['smoking_std'] = df['SMQ040'].apply(
                lambda x: 1.0 if x in [1, 2] else (0.0 if x == 3 else np.nan)
            )
        
        # 处理空腹状态
        if 'PHAFSTHR' in df.columns:
            fasting_hours = df['PHAFSTHR'].fillna(0)
            if 'PHAFSTMN' in df.columns:
                fasting_hours += df['PHAFSTMN'].fillna(0) / 60
            df['fasting_state_std'] = (fasting_hours >= 8).astype(float)
            df.loc[df['PHAFSTHR'].isna(), 'fasting_state_std'] = np.nan
        
        # 计算代谢综合征标签
        print("\n计算代谢综合征指标...")
        
        # 腰围异常
        waist_abn = 0
        if 'BMXWAIST' in df.columns and 'gender_std' in df.columns:
            male = df['gender_std'] == 0
            female = df['gender_std'] == 1
            waist_abn = (male & (df['BMXWAIST'] >= 102)) | (female & (df['BMXWAIST'] >= 88))
        
        # 甘油三酯异常
        tg_abn = (df['LBXTR'] >= 150) if 'LBXTR' in df.columns else 0
        
        # HDL异常
        hdl_abn = 0
        if 'LBDHDD' in df.columns and 'gender_std' in df.columns:
            male = df['gender_std'] == 0
            female = df['gender_std'] == 1
            hdl_abn = (male & (df['LBDHDD'] < 40)) | (female & (df['LBDHDD'] < 50))
        
        # 血压异常
        bp_abn = 0
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            bp_abn = (df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)
        
        # 血糖异常 (考虑strict_fasting选项)
        glu_abn = 0
        if 'LBXGLU' in df.columns:
            if strict_fasting and 'fasting_state_std' in df.columns:
                # 严格模式：仅空腹≥8h样本判定
                fast_ok = df['fasting_state_std'] == 1.0
                glu_abn = fast_ok & (df['LBXGLU'] >= 100)
                print("  使用严格空腹模式 (≥8h)")
            else:
                # 宽松模式：所有样本判定
                glu_abn = df['LBXGLU'] >= 100
                print("  使用宽松模式 (不限空腹)")
        
        # 计算评分
        score = (waist_abn.astype(int) + tg_abn.astype(int) + hdl_abn.astype(int) + 
                bp_abn.astype(int) + glu_abn.astype(int))
        df['metabolic_score'] = score
        df['metabolic_syndrome'] = (score >= 3).astype(int)
        
        # 定义特征列
        feature_cols = []
        
        # 核心特征
        core_map = {
            'BMXWAIST': 'waist_circumference',
            'systolic_bp': 'systolic_bp',
            'diastolic_bp': 'diastolic_bp',
            'LBXTR': 'triglycerides',
            'LBDHDD': 'hdl_cholesterol',
            'LBXGLU': 'fasting_glucose',
            'RIDAGEYR': 'age'
        }
        
        for old_name, new_name in core_map.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
                feature_cols.append(new_name)
        
        # 基础特征
        if 'BMXBMI' in df.columns:
            df['bmi'] = df['BMXBMI']
            feature_cols.append('bmi')
        
        if 'LBXGH' in df.columns:
            df['hba1c'] = df['LBXGH']
            feature_cols.append('hba1c')
        
        # 门控特征 (已标准化)
        gate_cols = ['gender_std', 'age_band', 'bp_med_std', 'lipid_med_std', 
                    'glucose_med_std', 'smoking_std', 'fasting_state_std']
        for col in gate_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # 核心特征缺失删除
        core_features = ['waist_circumference', 'systolic_bp', 'diastolic_bp',
                        'triglycerides', 'hdl_cholesterol', 'fasting_glucose', 'age']
        available_core = [c for c in core_features if c in df.columns]
        
        print("\n处理缺失值...")
        before_len = len(df)
        df = df.dropna(subset=available_core)
        print(f"  删除核心特征缺失: {before_len} → {len(df)}")
        
        # 非核心数值特征中位数填充
        for col in ['bmi', 'hba1c']:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  {col}: 中位数填充 ({median_val:.1f})")
        
        # 门控特征0填充
        for col in gate_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        # Winsorize选项
        if winsorize:
            print("\n应用Winsorize (0.1%, 99.9%)...")
            winsorize_cols = ['waist_circumference', 'systolic_bp', 'diastolic_bp',
                            'triglycerides', 'hdl_cholesterol', 'fasting_glucose']
            for col in winsorize_cols:
                if col in df.columns:
                    x = df[col]
                    lo, hi = x.quantile(0.001), x.quantile(0.999)
                    df[col] = x.clip(lo, hi)
        
        # 最终数据集
        final_cols = feature_cols + ['metabolic_score', 'metabolic_syndrome']
        final_df = df[final_cols].reset_index(drop=True)
        
        print(f"\n最终数据集: {len(final_df)} 行 × {len(final_df.columns)} 列")
        ms_rate = final_df['metabolic_syndrome'].mean() * 100
        print(f"代谢综合征患病率: {ms_rate:.1f}%")
        
        return final_df, feature_cols


def main():
    parser = argparse.ArgumentParser(description='NHANES XPT → LGU CSV v4.3')
    parser.add_argument('--data_dir', default='nhanes_data', 
                       help='XPT文件目录 (默认: nhanes_data)')
    parser.add_argument('--outdir', required=True, help='输出目录')
    parser.add_argument('--strict_fasting', action='store_true',
                       help='严格空腹模式：仅空腹≥8h判定血糖异常')
    parser.add_argument('--winsorize', action='store_true',
                       help='应用轻度winsorization (0.1%/99.9%)')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 处理数据
    processor = NHANESProcessor(args.data_dir)
    
    # 1. 加载XPT文件
    data = processor.load_xpt_files()
    if not data:
        print("\n错误：无法加载任何数据文件")
        return
    
    # 2. 合并数据集
    merged = processor.merge_datasets()
    if merged is None:
        print("\n错误：数据合并失败")
        return
    
    # 3. 标准化和处理
    result = processor.standardize_and_process(
        strict_fasting=args.strict_fasting,
        winsorize=args.winsorize
    )
    
    if result is None:
        print("\n错误：数据处理失败")
        return
    
    final_df, feature_cols = result
    
    # 4. 生成输出文件
    # 分离X和y
    X = final_df[feature_cols]
    
    # 回归任务CSV
    score_df = final_df[feature_cols + ['metabolic_score']].copy()
    score_csv = outdir / "NHANES_metabolic_score.csv"
    score_df.to_csv(score_csv, index=False)
    
    # 分类任务CSV
    syndrome_df = final_df[feature_cols + ['metabolic_syndrome']].copy()
    syndrome_csv = outdir / "NHANES_metabolic_syndrome.csv"
    syndrome_df.to_csv(syndrome_csv, index=False)
    
    # 数据字典
    dict_file = outdir / "nhanes_LGU_data_dictionary_v4_3.txt"
    data_dict = """NHANES LGU符号回归数据集 v4.3
=====================================

输入特征 (X)
------------
【核心特征 - MetS五项】
waist_circumference: 腰围 (cm)
systolic_bp: 收缩压 (mmHg) - 多次测量平均值
diastolic_bp: 舒张压 (mmHg) - 多次测量平均值  
triglycerides: 甘油三酯 (mg/dL)
hdl_cholesterol: HDL胆固醇 (mg/dL)
fasting_glucose: 空腹血糖 (mg/dL)

【基础特征】
age: 年龄 (岁)
bmi: 身体质量指数 (kg/m²)
hba1c: 糖化血红蛋白 (%)

【门控特征】
gender_std: 性别 (0=男性, 1=女性)
age_band: 年龄分段 (0=18-29, 1=30-39, 2=40-49, 3=50-59, 4=60-69, 5=70+)
bp_med_std: 降压药使用 (0=未使用, 1=使用)
lipid_med_std: 降脂药使用 (0=未使用, 1=使用)
glucose_med_std: 降糖药使用 (0=未使用, 1=使用)
smoking_std: 吸烟状态 (0=不吸烟, 1=吸烟)
fasting_state_std: 空腹状态 (0=非空腹<8h, 1=空腹≥8h)

目标变量 (y)
-----------
metabolic_score: 代谢综合征评分 (0-5, 异常组分数量) - 回归任务
metabolic_syndrome: 代谢综合征诊断 (0/1, ≥3项异常) - 分类任务

处理选项
--------
--strict_fasting: 严格空腹模式（仅空腹≥8h判定血糖异常）
--winsorize: 轻度尾部截断 (0.1%/99.9%)

注意事项
--------
1. 核心特征任一缺失则删除样本
2. BMI/HbA1c使用中位数填充
3. 门控特征缺失用0填充
4. 所有门控特征标准化为_std后缀
5. 性别统一编码: 0=男性, 1=女性

数据来源: 
NHANES 2017-2018
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2017
"""
    
    with open(dict_file, 'w', encoding='utf-8') as f:
        f.write(data_dict)
    
    # zscore元数据 (仅X)
    zscore_file = outdir / "zscore_meta.json"
    mu = X.mean(axis=0).to_dict()
    sd = X.std(axis=0, ddof=0).replace(0, 1.0).to_dict()
    zscore_meta = {"mu": mu, "sd": sd}
    
    with open(zscore_file, 'w', encoding='utf-8') as f:
        json.dump(zscore_meta, f, indent=2, ensure_ascii=False)
    
    # 输出总结
    print("\n" + "="*60)
    print("✔ 处理完成！")
    print("="*60)
    print(f"生成文件:")
    print(f"  - {score_csv.name}: {score_df.shape}")
    print(f"  - {syndrome_csv.name}: {syndrome_df.shape}")
    print(f"  - {dict_file.name}")
    print(f"  - {zscore_file.name}")
    # 额外输出：ground_truth.json（ATP III 阈值，原始临床单位）
    gt_file = outdir / "ground_truth.json"
    gt_payload = {
        "waist_circumference": {"values": [102.0, 88.0]},
        "triglycerides":       {"values": [150.0]},
        "hdl_cholesterol":     {"values": [40.0, 50.0]},
        "systolic_bp":         {"values": [130.0]},
        "diastolic_bp":        {"values": [85.0]},
        "fasting_glucose":     {"values": [100.0]}
    }
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(gt_payload, f, indent=2, ensure_ascii=False)
    print(f"  - {gt_file.name}")
    
    
    if args.strict_fasting:
        print(f"\n选项: 严格空腹模式 ✓")
    if args.winsorize:
        print(f"选项: Winsorize (0.1%/99.9%) ✓")
    
    # 显示特征摘要
    print(f"\n特征摘要:")
    print(f"  核心特征: {sum(1 for c in feature_cols if c in ['waist_circumference', 'systolic_bp', 'diastolic_bp', 'triglycerides', 'hdl_cholesterol', 'fasting_glucose', 'age'])} 个")
    print(f"  基础特征: {sum(1 for c in feature_cols if c in ['bmi', 'hba1c'])} 个")
    print(f"  门控特征: {sum(1 for c in feature_cols if '_std' in c or c == 'age_band')} 个")
    print(f"  总特征数: {len(feature_cols)} 个")
    
    # 显示前5行
    print(f"\n数据预览 (前5行):")
    print(score_df.head())


if __name__ == "__main__":
    main()