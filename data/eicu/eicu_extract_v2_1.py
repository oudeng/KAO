# eicu_extract_v2.py
"""
eICU ICU Composite Risk Score Dataset Extractor - Version 2.0

Purpose: Extract a dataset from eICU database with IMPROVED data completeness
         for key clinical variables (MAP, Lactate) to enable meaningful
         LGO threshold audit comparable to MIMIC-IV ICU results.

Key Improvements over v1:
1. SQL-level pre-filtering: Only select patients with MAP AND/OR Lactate measurements
2. Stratified sampling: Ensure representation across risk levels
3. Relaxed selection modes: Support different completeness requirements
4. Enhanced diagnostics: Report data availability at each step

Selection Modes:
- 'strict': Require BOTH MAP AND Lactate (highest quality, smaller sample)
- 'relaxed': Require MAP OR Lactate (balanced approach)  
- 'map_only': Require only MAP (for MAP-focused analysis)
- 'lactate_only': Require only Lactate (for Lactate-focused analysis)

Clinical Anchors for LGO Threshold Audit:
- MAP (mmHg): 65 mmHg (Sepsis-3/SSC)
- Lactate (mmol/L): 2.0 mmol/L (tissue hypoperfusion)
- Creatinine (mg/dL): 1.5 mg/dL (AKI Stage 1)
- SpO2 (%): 92% (hypoxemia)
- Respiratory Rate (/min): 24/min (tachypnea)

Author: Ou Deng
Version: 2.0
Date: December 2025
"""

import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class eICURiskScoreExtractorV2:
    """eICU Composite Risk Score Dataset Extractor - Version 2 with improved data completeness"""

    def __init__(self, db_config):
        """Initialize database connection configuration"""
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.extraction_stats = {}  # Track extraction statistics

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            self.cursor = self.conn.cursor()
            print("✓ Database connection successful!")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def check_data_availability(self, schema='eicu'):
        """
        Check availability of key variables in the database before extraction.
        This helps understand what's possible with the data.
        """
        print("\n" + "=" * 60)
        print("DATA AVAILABILITY CHECK")
        print("=" * 60)

        # Check total eligible patients (age >= 18, LOS >= 24h)
        query_base = f"""
        SELECT COUNT(DISTINCT patientunitstayid)
        FROM {schema}.patient
        WHERE age != '' AND age != '> 89'
        AND CAST(age AS INTEGER) >= 18
        AND unitdischargeoffset >= 1440
        """
        self.cursor.execute(query_base)
        total_eligible = self.cursor.fetchone()[0]
        print(f"\nTotal eligible patients (age≥18, LOS≥24h): {total_eligible:,}")

        # Check patients with MAP in first 24h
        query_map = f"""
        SELECT COUNT(DISTINCT p.patientunitstayid)
        FROM {schema}.patient p
        WHERE p.age != '' AND p.age != '> 89'
        AND CAST(p.age AS INTEGER) >= 18
        AND p.unitdischargeoffset >= 1440
        AND EXISTS (
            SELECT 1 FROM {schema}.vitalperiodic vp
            WHERE vp.patientunitstayid = p.patientunitstayid
            AND vp.observationoffset >= 0 AND vp.observationoffset <= 1440
            AND vp.systemicmean IS NOT NULL
            AND vp.systemicmean BETWEEN 30 AND 200
        )
        """
        self.cursor.execute(query_map)
        patients_with_map = self.cursor.fetchone()[0]
        print(
            f"Patients with valid MAP (first 24h): {patients_with_map:,} ({patients_with_map / total_eligible * 100:.1f}%)")

        # Check patients with Lactate in first 24h
        query_lactate = f"""
        SELECT COUNT(DISTINCT p.patientunitstayid)
        FROM {schema}.patient p
        WHERE p.age != '' AND p.age != '> 89'
        AND CAST(p.age AS INTEGER) >= 18
        AND p.unitdischargeoffset >= 1440
        AND EXISTS (
            SELECT 1 FROM {schema}.lab l
            WHERE l.patientunitstayid = p.patientunitstayid
            AND l.labresultoffset >= 0 AND l.labresultoffset <= 1440
            AND l.labname = 'lactate'
            AND l.labresult > 0 AND l.labresult < 50
        )
        """
        self.cursor.execute(query_lactate)
        patients_with_lactate = self.cursor.fetchone()[0]
        print(
            f"Patients with valid Lactate (first 24h): {patients_with_lactate:,} ({patients_with_lactate / total_eligible * 100:.1f}%)")

        # Check patients with BOTH MAP and Lactate
        query_both = f"""
        SELECT COUNT(DISTINCT p.patientunitstayid)
        FROM {schema}.patient p
        WHERE p.age != '' AND p.age != '> 89'
        AND CAST(p.age AS INTEGER) >= 18
        AND p.unitdischargeoffset >= 1440
        AND EXISTS (
            SELECT 1 FROM {schema}.vitalperiodic vp
            WHERE vp.patientunitstayid = p.patientunitstayid
            AND vp.observationoffset >= 0 AND vp.observationoffset <= 1440
            AND vp.systemicmean IS NOT NULL
            AND vp.systemicmean BETWEEN 30 AND 200
        )
        AND EXISTS (
            SELECT 1 FROM {schema}.lab l
            WHERE l.patientunitstayid = p.patientunitstayid
            AND l.labresultoffset >= 0 AND l.labresultoffset <= 1440
            AND l.labname = 'lactate'
            AND l.labresult > 0 AND l.labresult < 50
        )
        """
        self.cursor.execute(query_both)
        patients_with_both = self.cursor.fetchone()[0]
        print(
            f"Patients with BOTH MAP AND Lactate: {patients_with_both:,} ({patients_with_both / total_eligible * 100:.1f}%)")

        # Check patients with MAP OR Lactate
        query_either = f"""
        SELECT COUNT(DISTINCT p.patientunitstayid)
        FROM {schema}.patient p
        WHERE p.age != '' AND p.age != '> 89'
        AND CAST(p.age AS INTEGER) >= 18
        AND p.unitdischargeoffset >= 1440
        AND (
            EXISTS (
                SELECT 1 FROM {schema}.vitalperiodic vp
                WHERE vp.patientunitstayid = p.patientunitstayid
                AND vp.observationoffset >= 0 AND vp.observationoffset <= 1440
                AND vp.systemicmean IS NOT NULL
                AND vp.systemicmean BETWEEN 30 AND 200
            )
            OR EXISTS (
                SELECT 1 FROM {schema}.lab l
                WHERE l.patientunitstayid = p.patientunitstayid
                AND l.labresultoffset >= 0 AND l.labresultoffset <= 1440
                AND l.labname = 'lactate'
                AND l.labresult > 0 AND l.labresult < 50
            )
        )
        """
        self.cursor.execute(query_either)
        patients_with_either = self.cursor.fetchone()[0]
        print(
            f"Patients with MAP OR Lactate: {patients_with_either:,} ({patients_with_either / total_eligible * 100:.1f}%)")

        # Store stats
        self.extraction_stats['total_eligible'] = total_eligible
        self.extraction_stats['with_map'] = patients_with_map
        self.extraction_stats['with_lactate'] = patients_with_lactate
        self.extraction_stats['with_both'] = patients_with_both
        self.extraction_stats['with_either'] = patients_with_either

        print("\n" + "-" * 60)
        print("RECOMMENDATION:")
        if patients_with_both >= 3000:
            print(f"  → Use 'strict' mode: {patients_with_both:,} patients with BOTH variables")
        elif patients_with_either >= 3000:
            print(f"  → Use 'relaxed' mode: {patients_with_either:,} patients with EITHER variable")
        else:
            print(f"  → Limited data availability. Consider expanding time window or criteria.")
        print("-" * 60)

        return self.extraction_stats

    def get_icu_patients(self, n_samples=5000, schema='eicu', selection_mode='relaxed'):
        """
        Get ICU patient records with GUARANTEED key variable availability.

        Selection Modes:
        - 'strict': Require BOTH MAP AND Lactate measurements
        - 'relaxed': Require MAP OR Lactate measurements (default)
        - 'map_only': Require only MAP measurements
        - 'lactate_only': Require only Lactate measurements

        Selection criteria:
        - Adult patients (age >= 18)
        - ICU length of stay >= 24 hours
        - Key clinical variables available based on selection_mode
        """
        print(f"\nExtracting ICU patients (mode: {selection_mode})...")

        # Build the EXISTS clause based on selection mode
        if selection_mode == 'strict':
            exists_clause = """
            AND EXISTS (
                SELECT 1 FROM {schema}.vitalperiodic vp
                WHERE vp.patientunitstayid = p.patientunitstayid
                AND vp.observationoffset >= 0 AND vp.observationoffset <= 1440
                AND vp.systemicmean IS NOT NULL
                AND vp.systemicmean BETWEEN 30 AND 200
            )
            AND EXISTS (
                SELECT 1 FROM {schema}.lab l
                WHERE l.patientunitstayid = p.patientunitstayid
                AND l.labresultoffset >= 0 AND l.labresultoffset <= 1440
                AND l.labname = 'lactate'
                AND l.labresult > 0 AND l.labresult < 50
            )
            """.format(schema=schema)
        elif selection_mode == 'relaxed':
            exists_clause = """
            AND (
                EXISTS (
                    SELECT 1 FROM {schema}.vitalperiodic vp
                    WHERE vp.patientunitstayid = p.patientunitstayid
                    AND vp.observationoffset >= 0 AND vp.observationoffset <= 1440
                    AND vp.systemicmean IS NOT NULL
                    AND vp.systemicmean BETWEEN 30 AND 200
                )
                OR EXISTS (
                    SELECT 1 FROM {schema}.lab l
                    WHERE l.patientunitstayid = p.patientunitstayid
                    AND l.labresultoffset >= 0 AND l.labresultoffset <= 1440
                    AND l.labname = 'lactate'
                    AND l.labresult > 0 AND l.labresult < 50
                )
            )
            """.format(schema=schema)
        elif selection_mode == 'map_only':
            exists_clause = """
            AND EXISTS (
                SELECT 1 FROM {schema}.vitalperiodic vp
                WHERE vp.patientunitstayid = p.patientunitstayid
                AND vp.observationoffset >= 0 AND vp.observationoffset <= 1440
                AND vp.systemicmean IS NOT NULL
                AND vp.systemicmean BETWEEN 30 AND 200
            )
            """.format(schema=schema)
        elif selection_mode == 'lactate_only':
            exists_clause = """
            AND EXISTS (
                SELECT 1 FROM {schema}.lab l
                WHERE l.patientunitstayid = p.patientunitstayid
                AND l.labresultoffset >= 0 AND l.labresultoffset <= 1440
                AND l.labname = 'lactate'
                AND l.labresult > 0 AND l.labresult < 50
            )
            """.format(schema=schema)
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        query = f"""
        WITH patient_selection AS (
            SELECT 
                p.patientunitstayid,
                p.patienthealthsystemstayid,
                p.uniquepid,
                p.gender,
                p.age,
                p.unitdischargestatus,
                p.hospitaldischargestatus,
                p.unitdischargeoffset / 60.0 as los_hours,
                -- Mortality outcomes
                CASE WHEN p.unitdischargestatus = 'Expired' THEN 1 ELSE 0 END as icu_death,
                CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END as hospital_death
            FROM {schema}.patient p
            WHERE 
                -- Age filter: >=18 and <90 (eICU uses "> 89" for age)
                p.age != '' 
                AND p.age != '> 89'
                AND CAST(p.age AS INTEGER) >= 18
                -- LOS filter: at least 24 hours (1440 minutes)
                AND p.unitdischargeoffset >= 1440
                -- KEY VARIABLE AVAILABILITY FILTER
                {exists_clause}
        )
        SELECT *
        FROM patient_selection
        ORDER BY RANDOM()
        LIMIT {n_samples}
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        columns = ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid',
                   'gender', 'age', 'unitdischargestatus', 'hospitaldischargestatus',
                   'los_hours', 'icu_death', 'hospital_death']

        df = pd.DataFrame(results, columns=columns)

        # Convert age to numeric
        df['age_years'] = pd.to_numeric(df['age'], errors='coerce')

        print(f"✓ Found {len(df)} ICU patient records with required variables")
        print(f"  ICU mortality rate: {df['icu_death'].mean():.2%}")
        print(f"  Hospital mortality rate: {df['hospital_death'].mean():.2%}")

        self.extraction_stats['selected_patients'] = len(df)
        self.extraction_stats['selection_mode'] = selection_mode

        return df

    def extract_vital_signs(self, patient_ids, schema='eicu'):
        """
        Extract vital signs from vitalperiodic table (first 24 hours)
        """
        print("Extracting vital signs from first 24 hours...")

        patient_ids_str = ','.join(map(str, patient_ids))

        query = f"""
        WITH first_24h AS (
            SELECT 
                vp.patientunitstayid,
                vp.systemicmean,
                vp.systemicsystolic,
                vp.systemicdiastolic,
                vp.heartrate,
                vp.respiration,
                vp.sao2,
                vp.temperature
            FROM {schema}.vitalperiodic vp
            WHERE vp.patientunitstayid IN ({patient_ids_str})
            AND vp.observationoffset >= 0 
            AND vp.observationoffset <= 1440
        )
        SELECT 
            patientunitstayid,
            MIN(systemicmean) as map_min,
            AVG(systemicmean) as map_mean,
            MIN(systemicsystolic) as sbp_min,
            MIN(systemicdiastolic) as dbp_min,
            MAX(heartrate) as hr_max,
            AVG(heartrate) as hr_mean,
            MAX(respiration) as resprate_max,
            MIN(sao2) as spo2_min,
            MAX(temperature) as temp_max,
            MIN(temperature) as temp_min
        FROM first_24h
        GROUP BY patientunitstayid
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        columns = [
            'patientunitstayid', 'map_min', 'map_mean', 'sbp_min', 'dbp_min',
            'hr_max', 'hr_mean', 'resprate_max', 'spo2_min', 'temp_max', 'temp_min'
        ]

        df = pd.DataFrame(results, columns=columns)

        # Convert all numeric columns to numeric first
        numeric_cols = [
            'map_min', 'map_mean', 'sbp_min', 'dbp_min',
            'hr_max', 'hr_mean', 'resprate_max',
            'spo2_min', 'temp_max', 'temp_min'
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Safe clipping (NaN unaffected)
        df['map_min'] = df['map_min'].clip(lower=30, upper=200)
        df['map_mean'] = df['map_mean'].clip(lower=30, upper=200)
        df['sbp_min'] = df['sbp_min'].clip(lower=50, upper=250)
        df['dbp_min'] = df['dbp_min'].clip(lower=20, upper=150)
        df['hr_max'] = df['hr_max'].clip(lower=30, upper=250)
        df['hr_mean'] = df['hr_mean'].clip(lower=30, upper=250)
        df['resprate_max'] = df['resprate_max'].clip(lower=5, upper=60)
        df['spo2_min'] = df['spo2_min'].clip(lower=50, upper=100)
        df['temp_max'] = df['temp_max'].clip(lower=32, upper=45)
        df['temp_min'] = df['temp_min'].clip(lower=32, upper=45)

        # Report MAP availability
        map_available = df['map_min'].notna().sum()
        print(f"  ✓ MAP available: {map_available}/{len(df)} ({map_available / len(df) * 100:.1f}%)")

        return df

    def extract_labs(self, patient_ids, schema='eicu'):
        """
        Extract laboratory values from lab table (first 24 hours)
        """
        print("Extracting laboratory values from first 24 hours...")

        patient_ids_str = ','.join(map(str, patient_ids))

        query = f"""
        WITH first_24h_labs AS (
            SELECT 
                l.patientunitstayid,
                l.labname,
                l.labresult
            FROM {schema}.lab l
            WHERE l.patientunitstayid IN ({patient_ids_str})
            AND l.labresultoffset >= 0 
            AND l.labresultoffset <= 1440
            AND l.labresult IS NOT NULL
        )
        SELECT 
            patientunitstayid,
            -- Lactate (key threshold variable)
            MAX(CASE WHEN labname = 'lactate' AND labresult > 0 AND labresult < 50 
                     THEN labresult END) as lactate_max,
            AVG(CASE WHEN labname = 'lactate' AND labresult > 0 AND labresult < 50 
                     THEN labresult END) as lactate_mean,

            -- Creatinine (key threshold variable)
            MAX(CASE WHEN labname = 'creatinine' AND labresult > 0 AND labresult < 30 
                     THEN labresult END) as creatinine_max,

            -- Sodium
            MIN(CASE WHEN labname = 'sodium' AND labresult > 100 AND labresult < 180 
                     THEN labresult END) as sodium_min,
            MAX(CASE WHEN labname = 'sodium' AND labresult > 100 AND labresult < 180 
                     THEN labresult END) as sodium_max,

            -- Hemoglobin
            MIN(CASE WHEN labname = 'Hgb' AND labresult > 0 AND labresult < 25 
                     THEN labresult END) as hemoglobin_min,

            -- WBC
            MAX(CASE WHEN labname = 'WBC x 1000' AND labresult > 0 AND labresult < 100 
                     THEN labresult END) as wbc_max,

            -- Platelets
            MIN(CASE WHEN labname = 'platelets x 1000' AND labresult > 0 AND labresult < 1500 
                     THEN labresult END) as platelets_min,

            -- BUN
            MAX(CASE WHEN labname = 'BUN' AND labresult > 0 AND labresult < 300 
                     THEN labresult END) as bun_max,

            -- Glucose
            MAX(CASE WHEN labname = 'glucose' AND labresult > 0 AND labresult < 1500 
                     THEN labresult END) as glucose_max,
            MIN(CASE WHEN labname = 'glucose' AND labresult > 0 AND labresult < 1500 
                     THEN labresult END) as glucose_min,

            -- Bilirubin
            MAX(CASE WHEN labname = 'total bilirubin' AND labresult >= 0 AND labresult < 50 
                     THEN labresult END) as bilirubin_max,

            -- pH (arterial blood gas)
            MIN(CASE WHEN labname = 'pH' AND labresult > 6.5 AND labresult < 8.0 
                     THEN labresult END) as ph_min,

            -- PaO2
            MIN(CASE WHEN labname = 'paO2' AND labresult > 0 AND labresult < 700 
                     THEN labresult END) as pao2_min,

            -- PaCO2
            MAX(CASE WHEN labname = 'paCO2' AND labresult > 0 AND labresult < 200 
                     THEN labresult END) as paco2_max,

            -- FiO2
            MAX(CASE WHEN labname = 'FiO2' AND labresult > 0 AND labresult <= 100 
                     THEN labresult END) as fio2_max
        FROM first_24h_labs
        GROUP BY patientunitstayid
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        columns = ['patientunitstayid', 'lactate_max', 'lactate_mean', 'creatinine_max',
                   'sodium_min', 'sodium_max', 'hemoglobin_min', 'wbc_max', 'platelets_min',
                   'bun_max', 'glucose_max', 'glucose_min', 'bilirubin_max',
                   'ph_min', 'pao2_min', 'paco2_max', 'fio2_max']

        df = pd.DataFrame(results, columns=columns)

        # Report Lactate availability
        lactate_available = df['lactate_max'].notna().sum()
        print(f"  ✓ Lactate available: {lactate_available}/{len(df)} ({lactate_available / len(df) * 100:.1f}%)")

        return df

    def extract_apache_variables(self, patient_ids, schema='eicu'):
        """Extract APACHE severity variables"""
        print("Extracting APACHE variables...")

        patient_ids_str = ','.join(map(str, patient_ids))

        query = f"""
        SELECT 
            patientunitstayid,
            verbal as gcs_verbal,
            motor as gcs_motor,
            eyes as gcs_eyes,
            COALESCE(verbal, 1) + COALESCE(motor, 1) + COALESCE(eyes, 1) as gcs_total,
            intubated,
            vent,
            dialysis
        FROM {schema}.apacheapsvar
        WHERE patientunitstayid IN ({patient_ids_str})
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        columns = ['patientunitstayid', 'gcs_verbal', 'gcs_motor', 'gcs_eyes',
                   'gcs_total', 'intubated', 'vent', 'dialysis']

        df = pd.DataFrame(results, columns=columns)
        df['gcs_total'] = df['gcs_total'].clip(3, 15)

        print(f"  ✓ GCS available: {df['gcs_total'].notna().sum()}/{len(df)}")
        return df

    def extract_vasopressor_use(self, patient_ids, schema='eicu'):
        """Extract vasopressor/inotrope usage"""
        print("Extracting vasopressor usage...")

        patient_ids_str = ','.join(map(str, patient_ids))

        vasopressor_drugs = [
            'Norepinephrine', 'norepinephrine', 'Levophed',
            'Dopamine', 'dopamine',
            'Epinephrine', 'epinephrine', 'Adrenalin',
            'Vasopressin', 'vasopressin',
            'Phenylephrine', 'phenylephrine', 'Neo-Synephrine',
            'Dobutamine', 'dobutamine'
        ]

        vasopressor_pattern = '|'.join(vasopressor_drugs)

        query = f"""
        WITH vasopressor_usage AS (
            SELECT 
                patientunitstayid,
                drugname,
                CASE 
                    WHEN drugrate ~ '^[0-9]+\\.?[0-9]*$' THEN CAST(drugrate AS NUMERIC)
                    ELSE NULL
                END as drugrate_numeric,
                infusionoffset
            FROM {schema}.infusiondrug
            WHERE patientunitstayid IN ({patient_ids_str})
            AND infusionoffset >= 0 
            AND infusionoffset <= 1440
            AND drugrate IS NOT NULL
            AND (drugname ~* '{vasopressor_pattern}')
        )
        SELECT 
            patientunitstayid,
            1 as vasopressor_use,
            MAX(drugrate_numeric) as vasopressor_rate_max,
            COUNT(DISTINCT drugname) as vasopressor_count
        FROM vasopressor_usage
        WHERE drugrate_numeric IS NOT NULL AND drugrate_numeric > 0
        GROUP BY patientunitstayid
        """

        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            columns = ['patientunitstayid', 'vasopressor_use', 'vasopressor_rate_max', 'vasopressor_count']
            df = pd.DataFrame(results, columns=columns)
            print(f"  ✓ Vasopressor use: {len(df)} patients")
            return df
        except Exception as e:
            print(f"  ✗ Error extracting vasopressor usage: {e}")
            return pd.DataFrame(
                columns=['patientunitstayid', 'vasopressor_use', 'vasopressor_rate_max', 'vasopressor_count'])

    def extract_mechanical_ventilation(self, patient_ids, schema='eicu'):
        """Extract mechanical ventilation status"""
        print("Extracting mechanical ventilation status...")

        patient_ids_str = ','.join(map(str, patient_ids))

        query = f"""
        WITH mv_status AS (
            SELECT DISTINCT patientunitstayid, 1 as on_vent
            FROM {schema}.respiratorycare
            WHERE patientunitstayid IN ({patient_ids_str})
            AND ventstartoffset <= 1440
            AND airwaytype IS NOT NULL

            UNION

            SELECT DISTINCT patientunitstayid, 1 as on_vent
            FROM {schema}.respiratorycharting
            WHERE patientunitstayid IN ({patient_ids_str})
            AND respchartoffset >= 0 
            AND respchartoffset <= 1440
        )
        SELECT patientunitstayid, MAX(on_vent) as mechanical_ventilation
        FROM mv_status
        GROUP BY patientunitstayid
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        columns = ['patientunitstayid', 'mechanical_ventilation']
        df = pd.DataFrame(results, columns=columns)

        print(f"  ✓ Mechanical ventilation: {len(df)} patients")
        return df

    def extract_urine_output(self, patient_ids, schema='eicu'):
        """Extract urine output"""
        print("Extracting urine output...")

        patient_ids_str = ','.join(map(str, patient_ids))

        query = f"""
        SELECT 
            patientunitstayid,
            SUM(CASE WHEN cellvaluenumeric > 0 THEN cellvaluenumeric ELSE 0 END) as urine_output_24h
        FROM {schema}.intakeoutput
        WHERE patientunitstayid IN ({patient_ids_str})
        AND intakeoutputoffset >= 0 
        AND intakeoutputoffset <= 1440
        AND celllabel LIKE '%%urine%%' OR celllabel LIKE '%%Urine%%'
        GROUP BY patientunitstayid
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        columns = ['patientunitstayid', 'urine_output_24h']
        df = pd.DataFrame(results, columns=columns)

        print(f"  ✓ Urine output: {len(df)} patients")
        return df

    def combine_features(self, patients_df):
        """Combine all features into a single dataset"""
        print("\n" + "-" * 40)
        print("COMBINING ALL FEATURES")
        print("-" * 40)

        patient_ids = patients_df['patientunitstayid'].tolist()

        # Extract all components
        vitals = self.extract_vital_signs(patient_ids)
        labs = self.extract_labs(patient_ids)
        apache = self.extract_apache_variables(patient_ids)
        vasopressors = self.extract_vasopressor_use(patient_ids)
        mv = self.extract_mechanical_ventilation(patient_ids)
        urine = self.extract_urine_output(patient_ids)

        # Merge all dataframes
        df = patients_df.copy()

        for feature_df in [vitals, labs, apache, vasopressors, mv, urine]:
            if not feature_df.empty:
                df = df.merge(feature_df, on='patientunitstayid', how='left')

        # Fill binary features only
        df['vasopressor_use'] = df['vasopressor_use'].fillna(0)
        df['mechanical_ventilation'] = df['mechanical_ventilation'].fillna(0)

        # Do NOT fill continuous clinical variables - preserve missingness for transparency

        print(f"\n✓ Combined dataset: {len(df)} samples, {len(df.columns)} features")
        return df

    def create_threshold_features(self, df):
        """Create binary threshold features for clinical interpretation"""
        print("\nCreating threshold features...")

        # MAP thresholds
        if 'map_min' in df.columns:
            df['map_below_65'] = (df['map_min'] < 65).astype(int)
            df['map_below_60'] = (df['map_min'] < 60).astype(int)

        # Lactate thresholds
        if 'lactate_max' in df.columns:
            df['lactate_elevated'] = (df['lactate_max'] > 2.0).astype(int)
            df['lactate_severely_elevated'] = (df['lactate_max'] > 4.0).astype(int)

        # Creatinine thresholds
        if 'creatinine_max' in df.columns:
            df['aki_stage1'] = (df['creatinine_max'] > 1.5).astype(int)
            df['aki_stage2'] = (df['creatinine_max'] > 2.0).astype(int)
            df['aki_stage3'] = (df['creatinine_max'] > 3.0).astype(int)

        # SpO2 threshold
        if 'spo2_min' in df.columns:
            df['hypoxemia'] = (df['spo2_min'] < 92).astype(int)
            df['severe_hypoxemia'] = (df['spo2_min'] < 88).astype(int)

        # Respiratory rate threshold
        if 'resprate_max' in df.columns:
            df['tachypnea'] = (df['resprate_max'] > 24).astype(int)

        # Heart rate threshold
        if 'hr_max' in df.columns:
            df['tachycardia'] = (df['hr_max'] > 100).astype(int)

        # GCS threshold
        if 'gcs_total' in df.columns:
            df['gcs_severe'] = (df['gcs_total'] < 8).astype(int)
            df['gcs_moderate'] = ((df['gcs_total'] >= 8) & (df['gcs_total'] < 13)).astype(int)

        # Age bands
        if 'age_years' in df.columns:
            df['elderly'] = (df['age_years'] > 65).astype(int)
            df['very_elderly'] = (df['age_years'] > 80).astype(int)

        # P/F ratio
        if 'pao2_min' in df.columns and 'fio2_max' in df.columns:
            def adjust_fio2(x):
                if pd.isna(x) or x is None:
                    return np.nan
                return x / 100 if x > 1 else x

            fio2_adjusted = df['fio2_max'].apply(adjust_fio2)
            df['pf_ratio'] = df['pao2_min'] / fio2_adjusted.replace(0, np.nan)
            df['ards_mild'] = (df['pf_ratio'] < 300).fillna(0).astype(int)
            df['ards_moderate'] = (df['pf_ratio'] < 200).fillna(0).astype(int)
            df['ards_severe'] = (df['pf_ratio'] < 100).fillna(0).astype(int)

        # Shock index
        if 'hr_max' in df.columns and 'sbp_min' in df.columns:
            sbp_safe = df['sbp_min'].replace(0, np.nan)
            df['shock_index'] = df['hr_max'] / sbp_safe
            df['shock_index_elevated'] = (df['shock_index'] > 0.9).fillna(0).astype(int)

        return df

    def calculate_composite_risk_score(self, df):
        """Calculate composite risk score based on threshold features"""
        print("\nCalculating composite risk score...")

        score = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

        # Hemodynamic components
        if 'map_below_65' in df.columns:
            score += df['map_below_65'].fillna(0).astype(int) * 1

        if 'lactate_elevated' in df.columns:
            score += df['lactate_elevated'].fillna(0).astype(int) * 2
        if 'lactate_severely_elevated' in df.columns:
            score += df['lactate_severely_elevated'].fillna(0).astype(int) * 1

        # Renal component
        if 'aki_stage1' in df.columns:
            score += df['aki_stage1'].fillna(0).astype(int) * 1
        if 'aki_stage2' in df.columns:
            score += df['aki_stage2'].fillna(0).astype(int) * 1

        # Respiratory component
        if 'hypoxemia' in df.columns:
            score += df['hypoxemia'].fillna(0).astype(int) * 1
        if 'mechanical_ventilation' in df.columns:
            score += df['mechanical_ventilation'].fillna(0).astype(int) * 2

        # Circulatory support
        if 'vasopressor_use' in df.columns:
            score += df['vasopressor_use'].fillna(0).astype(int) * 2

        # Neurological component
        if 'gcs_severe' in df.columns:
            score += df['gcs_severe'].fillna(0).astype(int) * 2

        # Vital sign abnormalities
        if 'tachycardia' in df.columns:
            score += df['tachycardia'].fillna(0).astype(int) * 1
        if 'tachypnea' in df.columns:
            score += df['tachypnea'].fillna(0).astype(int) * 1

        df['composite_risk_score'] = score

        print(f"  Score range: {score.min():.0f} - {score.max():.0f}")
        print(f"  Mean: {score.mean():.2f}, Median: {score.median():.0f}, Std: {score.std():.2f}")

        return df

    def prepare_final_dataset(self, df):
        """Prepare final dataset for LGO experiments"""
        print("\nPreparing final dataset...")

        # Primary features - aligned with MIMIC-IV format
        primary_columns = {
            'map_min': 'map_mmhg',
            'lactate_max': 'lactate_mmol_l',
            'creatinine_max': 'creatinine_mg_dl',
            'age_years': 'age_years',
            'gcs_total': 'gcs',
            'vasopressor_rate_max': 'vasopressor_dose',
            'sbp_min': 'sbp_min',
            'dbp_min': 'dbp_min',
            'hr_max': 'hr_max',
            'resprate_max': 'resprate_max',
            'spo2_min': 'spo2_min',
            'hemoglobin_min': 'hemoglobin_min',
            'sodium_min': 'sodium_min',
            'urine_output_24h': 'urine_output_min',
            'mechanical_ventilation': 'mechanical_ventilation_std',
            'vasopressor_use': 'vasopressor_use_std'
        }

        # Gender encoding
        df['gender_std'] = df['gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0.5)

        # Age band
        df['age_band'] = pd.cut(df['age_years'],
                                bins=[0, 30, 45, 60, 75, 100],
                                labels=[0, 1, 2, 3, 4]).astype(float)

        # Set hours_since_admission
        df['hours_since_admission'] = 24

        # Rename columns
        final_df = df.copy()
        for old_name, new_name in primary_columns.items():
            if old_name in final_df.columns:
                final_df = final_df.rename(columns={old_name: new_name})

        # Select final columns
        final_column_order = [
            'map_mmhg', 'lactate_mmol_l', 'creatinine_mg_dl', 'age_years', 'gcs',
            'vasopressor_dose', 'hours_since_admission', 'sbp_min', 'dbp_min',
            'hr_max', 'resprate_max', 'spo2_min', 'hemoglobin_min', 'sodium_min',
            'urine_output_min', 'mechanical_ventilation_std', 'vasopressor_use_std',
            'age_band', 'gender_std', 'composite_risk_score'
        ]

        available_columns = [c for c in final_column_order if c in final_df.columns]
        final_df = final_df[available_columns]

        print(f"✓ Final dataset: {len(final_df)} samples, {len(final_df.columns)} features")

        return final_df

    def analyze_dataset(self, df):
        """Analyze and report dataset quality metrics"""
        print("\n" + "=" * 60)
        print("DATASET QUALITY ANALYSIS")
        print("=" * 60)

        print(f"\nTotal samples: {len(df)}")
        print(f"Total features: {len(df.columns)}")

        # Target distribution
        if 'composite_risk_score' in df.columns:
            print(f"\nComposite Risk Score Distribution:")
            print(f"  Range: {df['composite_risk_score'].min():.0f} - {df['composite_risk_score'].max():.0f}")
            print(f"  Mean: {df['composite_risk_score'].mean():.2f}")
            print(f"  Median: {df['composite_risk_score'].median():.0f}")
            print(f"  Std: {df['composite_risk_score'].std():.2f}")

        # Key threshold variables - CRITICAL FOR LGO
        print(f"\n" + "-" * 40)
        print("KEY VARIABLES FOR LGO THRESHOLD AUDIT")
        print("-" * 40)
        threshold_vars = {
            'map_mmhg': ('MAP', 'mmHg', 65, '<'),
            'lactate_mmol_l': ('Lactate', 'mmol/L', 2.0, '>'),
            'creatinine_mg_dl': ('Creatinine', 'mg/dL', 1.5, '>'),
            'spo2_min': ('SpO2', '%', 92, '<'),
            'resprate_max': ('Resp Rate', '/min', 24, '>'),
            'hr_max': ('Heart Rate', 'bpm', 100, '>'),
            'gcs': ('GCS', '', 8, '<')
        }

        print(f"\n{'Variable':<20} {'Available':<15} {'Mean±Std':<20} {'Anchor':<15}")
        print("-" * 70)

        for var, (name, unit, threshold, direction) in threshold_vars.items():
            if var in df.columns:
                # 修复：将 Decimal 类型转换为 float，避免 std() 计算报错
                # FIX: Convert Decimal types to float to prevent std() calculation errors
                df[var] = pd.to_numeric(df[var], errors='coerce')

                available = df[var].notna().sum()
                pct = available / len(df) * 100
                mean_val = df[var].mean()
                std_val = df[var].std()

                if direction == '<':
                    below = (df[var] < threshold).mean() * 100
                    anchor_info = f"<{threshold} {unit}: {below:.1f}%"
                else:
                    above = (df[var] > threshold).mean() * 100
                    anchor_info = f">{threshold} {unit}: {above:.1f}%"

                status = "✓" if pct >= 50 else "⚠" if pct >= 20 else "✗"
                print(
                    f"{status} {name:<18} {available:>5} ({pct:>5.1f}%)    {mean_val:>6.1f}±{std_val:<6.1f}    {anchor_info}")

        # Missing data summary
        print(f"\n" + "-" * 40)
        print("MISSING DATA SUMMARY")
        print("-" * 40)
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
        missing_df = missing_df[missing_df['Count'] > 0].sort_values('Percent', ascending=False)

        if len(missing_df) > 0:
            for col, row in missing_df.iterrows():
                status = "✓" if row['Percent'] < 20 else "⚠" if row['Percent'] < 50 else "✗"
                print(f"  {status} {col}: {int(row['Count'])} ({row['Percent']:.1f}%)")
        else:
            print("  ✓ No missing values!")

        # Store final stats
        self.extraction_stats['final_samples'] = len(df)
        self.extraction_stats['map_available_pct'] = df[
                                                         'map_mmhg'].notna().mean() * 100 if 'map_mmhg' in df.columns else 0
        self.extraction_stats['lactate_available_pct'] = df[
                                                             'lactate_mmol_l'].notna().mean() * 100 if 'lactate_mmol_l' in df.columns else 0

        return df

    def export_dataset(self, df, filename='eICU_composite_risk_score_v2.csv'):
        """Export final dataset to CSV"""
        print(f"\n" + "=" * 60)
        print(f"EXPORTING DATASET")
        print("=" * 60)

        # Round numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['composite_risk_score', 'gcs', 'age_band', 'hours_since_admission']:
                df[col] = df[col].round(0).astype('Int64')
            else:
                df[col] = df[col].round(4)

        df.to_csv(filename, index=False)
        print(f"✓ Dataset saved: {filename}")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(df.columns)}")

        # Create data dictionary
        self._create_data_dictionary(df, filename.replace('.csv', '_dictionary.txt'))

        # Create extraction report
        self._create_extraction_report(filename.replace('.csv', '_extraction_report.txt'))

        return df

    def _create_data_dictionary(self, df, filename):
        """Create data dictionary"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("eICU Composite Risk Score Dataset (v2) - Data Dictionary\n")
            f.write("=" * 60 + "\n\n")

            f.write("PURPOSE:\n")
            f.write("-" * 40 + "\n")
            f.write("Dataset for LGO symbolic regression with improved data completeness.\n")
            f.write("Designed for threshold audit comparable to MIMIC-IV ICU results.\n\n")

            f.write("KEY IMPROVEMENTS (v2):\n")
            f.write("-" * 40 + "\n")
            f.write("1. SQL-level pre-filtering for key variable availability\n")
            f.write("2. Guaranteed MAP and/or Lactate measurements\n")
            f.write("3. Reduced missing rates for threshold audit variables\n\n")

            f.write("CLINICAL ANCHORS FOR LGO EVALUATION:\n")
            f.write("-" * 40 + "\n")
            f.write("Feature            | Unit      | Anchor Value | Source\n")
            f.write("-" * 60 + "\n")
            f.write("MAP               | mmHg      | 65          | Sepsis-3/SSC\n")
            f.write("Lactate           | mmol/L    | 2.0         | Sepsis-3\n")
            f.write("Creatinine        | mg/dL     | 1.5         | KDIGO AKI\n")
            f.write("SpO2              | %         | 92          | ARDS Network\n")
            f.write("Respiratory Rate  | /min      | 24          | qSOFA\n")
            f.write("Heart Rate        | bpm       | 100         | SIRS/qSOFA\n")
            f.write("GCS               | -         | 8           | TBI Guidelines\n\n")

            f.write("VARIABLE DESCRIPTIONS:\n")
            f.write("-" * 40 + "\n\n")

            for col in df.columns:
                f.write(f"{col}:\n")
                if col in ['map_mmhg']:
                    f.write("  Mean Arterial Pressure minimum (mmHg)\n")
                    f.write("  Clinical threshold: <65 mmHg (hypotension)\n")
                elif col in ['lactate_mmol_l']:
                    f.write("  Lactate maximum (mmol/L)\n")
                    f.write("  Clinical threshold: >2.0 mmol/L (tissue hypoperfusion)\n")
                elif col in ['creatinine_mg_dl']:
                    f.write("  Creatinine maximum (mg/dL)\n")
                    f.write("  Clinical threshold: >1.5 mg/dL (AKI Stage 1)\n")
                elif col in ['resprate_max']:
                    f.write("  Respiratory Rate maximum (breaths/min)\n")
                    f.write("  Clinical threshold: >24/min (tachypnea)\n")
                elif col in ['composite_risk_score']:
                    f.write("  Calculated composite risk score (target variable)\n")
                f.write("\n")

        print(f"✓ Data dictionary saved: {filename}")

    def _create_extraction_report(self, filename):
        """Create extraction report with statistics"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("eICU Dataset Extraction Report (v2)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("EXTRACTION STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.extraction_stats.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.2f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

            f.write("\n")
            f.write("COMPARISON WITH v1:\n")
            f.write("-" * 40 + "\n")
            f.write("v1 Issue: MAP missing 73.6%, Lactate missing 72.8%\n")
            f.write(f"v2 Result: MAP available {self.extraction_stats.get('map_available_pct', 0):.1f}%, ")
            f.write(f"Lactate available {self.extraction_stats.get('lactate_available_pct', 0):.1f}%\n")

        print(f"✓ Extraction report saved: {filename}")


def main():
    """Main function to extract eICU dataset with improved completeness"""

    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'eicu',
        'user': 'postgres',
        'password': '1233214'  # Modify as needed
    }

    # Extraction parameters
    N_SAMPLES = 5000
    SELECTION_MODE = 'strict'  # Options: 'strict', 'relaxed', 'map_only', 'lactate_only'

    # Create extractor
    extractor = eICURiskScoreExtractorV2(db_config)

    try:
        # Connect to database
        if not extractor.connect():
            print("Failed to connect to database. Please check configuration.")
            return

        print("=" * 60)
        print("eICU Composite Risk Score Dataset Extraction (v2)")
        print("Improved Data Completeness for LGO Threshold Audit")
        print("=" * 60)

        # 1. Check data availability first
        stats = extractor.check_data_availability()

        # Adjust selection mode based on availability
        if stats['with_both'] < 1000 and SELECTION_MODE == 'strict':
            print(f"\n⚠ Warning: Only {stats['with_both']} patients with BOTH variables.")
            print("  Consider using 'relaxed' mode instead.")
            user_input = input("  Continue with 'strict' mode? (y/n): ")
            if user_input.lower() != 'y':
                SELECTION_MODE = 'relaxed'
                print(f"  → Switching to '{SELECTION_MODE}' mode")

        # 2. Get ICU patients with key variable availability
        patients = extractor.get_icu_patients(n_samples=N_SAMPLES, selection_mode=SELECTION_MODE)

        if patients.empty:
            print("No patients found matching criteria")
            return

        # 3. Combine all features
        dataset = extractor.combine_features(patients)

        if dataset.empty:
            print("Feature extraction failed")
            return

        # 4. Create threshold features
        dataset = extractor.create_threshold_features(dataset)

        # 5. Calculate composite risk score
        dataset = extractor.calculate_composite_risk_score(dataset)

        # 6. Prepare final dataset
        final_dataset = extractor.prepare_final_dataset(dataset)

        # 7. Analyze dataset quality
        final_dataset = extractor.analyze_dataset(final_dataset)

        # 8. Export dataset
        output_filename = f'eICU_composite_risk_score_v2_{SELECTION_MODE}.csv'
        extractor.export_dataset(final_dataset, output_filename)

        print("\n" + "=" * 60)
        print("✓ Dataset extraction completed successfully!")
        print("=" * 60)

        # Final summary
        print(f"\nFINAL SUMMARY:")
        print(f"  Selection mode: {SELECTION_MODE}")
        print(f"  Samples extracted: {len(final_dataset)}")
        if 'map_mmhg' in final_dataset.columns:
            print(
                f"  MAP available: {final_dataset['map_mmhg'].notna().sum()} ({final_dataset['map_mmhg'].notna().mean() * 100:.1f}%)")
        if 'lactate_mmol_l' in final_dataset.columns:
            print(
                f"  Lactate available: {final_dataset['lactate_mmol_l'].notna().sum()} ({final_dataset['lactate_mmol_l'].notna().mean() * 100:.1f}%)")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

    finally:
        extractor.close()


if __name__ == "__main__":
    main()
