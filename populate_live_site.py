#!/usr/bin/env python3
"""
Populate the live website (https://www.hepatiq.site) directly through its authenticated REST API.
This populates:
1. The 3 Doctor accounts (dr_sarah, dr_omar, dr_layla) with full permissions.
2. The 25 diverse clinical patient cases with complete patient records.
3. Runs the live AI DiagnosisEngine for each patient, generating real clinical ML analyses and lab tests.
"""

import sys
import json
import time
import random
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_URL = "https://www.hepatiq.site/api"

# 25 Patient Definitions
PATIENTS_DATA = [
    # --- HEPATITIS C (7 Cases) ---
    {
        "name": "أحمد خالد المنصور",
        "patient_id": "PID-2026-101",
        "age": 54, "gender": 1,
        "doc_idx": 0, # dr_sarah
        "dept": "Hepatology",
        "profile": {
            'age': 54, 'gender': 1, 'bmi': 27.2, 'smoking': 'Yes', 'alcohol': 'Moderate',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'Medium',
            'ascites': 'Yes', 'hepatomegaly': 'Yes', 'spiders': 'Yes', 'edema': 'Moderate',
            'bilirubin': 3.8, 'albumin': 2.9, 'alp': 175, 'alt': 135, 'ast': 160,
            'platelets': 110000, 'prothrombin': 17.5, 'copper': 175, 'cholesterol': 155,
            'creatinine': 1.3, 'glucose': 125, 'ggt': 190, 'triglycerides': 145,
            'uric_acid': 7.2, 'hdl': 36, 'total_proteins': 6.4
        }
    },
    {
        "name": "مريم حسن القحطاني",
        "patient_id": "PID-2026-102",
        "age": 49, "gender": 0,
        "doc_idx": 1, # dr_omar
        "dept": "Oncology",
        "profile": {
            'age': 49, 'gender': 0, 'bmi': 25.1, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Moderate', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'Yes', 'spiders': 'No', 'edema': 'Slight',
            'bilirubin': 2.1, 'albumin': 3.4, 'alp': 130, 'alt': 88, 'ast': 95,
            'platelets': 165000, 'prothrombin': 14.5, 'copper': 140, 'cholesterol': 170,
            'creatinine': 1.0, 'glucose': 105, 'ggt': 120, 'triglycerides': 130,
            'uric_acid': 6.1, 'hdl': 42, 'total_proteins': 6.8
        }
    },
    {
        "name": "عبد الله سعود الشمري",
        "patient_id": "PID-2026-103",
        "age": 61, "gender": 1,
        "doc_idx": 2, # dr_layla
        "dept": "Gastroenterology",
        "profile": {
            'age': 61, 'gender': 1, 'bmi': 26.8, 'smoking': 'Yes', 'alcohol': 'Heavy',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'High',
            'ascites': 'Yes', 'hepatomegaly': 'Yes', 'spiders': 'Yes', 'edema': 'Severe',
            'bilirubin': 4.5, 'albumin': 2.6, 'alp': 210, 'alt': 160, 'ast': 195,
            'platelets': 95000, 'prothrombin': 19.2, 'copper': 190, 'cholesterol': 140,
            'creatinine': 1.5, 'glucose': 140, 'ggt': 230, 'triglycerides': 160,
            'uric_acid': 7.9, 'hdl': 32, 'total_proteins': 6.1
        }
    },
    {
        "name": "زينب علي باقر",
        "patient_id": "PID-2026-104",
        "age": 43, "gender": 0,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 43, 'gender': 0, 'bmi': 24.3, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Moderate', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 1.6, 'albumin': 3.7, 'alp': 105, 'alt': 65, 'ast': 58,
            'platelets': 210000, 'prothrombin': 13.2, 'copper': 125, 'cholesterol': 185,
            'creatinine': 0.9, 'glucose': 98, 'ggt': 75, 'triglycerides': 125,
            'uric_acid': 5.4, 'hdl': 48, 'total_proteins': 7.1
        }
    },
    {
        "name": "يوسف سالم السويدي",
        "patient_id": "PID-2026-105",
        "age": 57, "gender": 1,
        "doc_idx": 1,
        "dept": "Oncology",
        "profile": {
            'age': 57, 'gender': 1, 'bmi': 28.4, 'smoking': 'Yes', 'alcohol': 'Moderate',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'Medium',
            'ascites': 'Slight', 'hepatomegaly': 'Yes', 'spiders': 'Yes', 'edema': 'Moderate',
            'bilirubin': 2.9, 'albumin': 3.1, 'alp': 160, 'alt': 110, 'ast': 125,
            'platelets': 135000, 'prothrombin': 16.0, 'copper': 160, 'cholesterol': 165,
            'creatinine': 1.2, 'glucose': 118, 'ggt': 170, 'triglycerides': 150,
            'uric_acid': 6.8, 'hdl': 38, 'total_proteins': 6.6
        }
    },
    {
        "name": "طارق ناصر العمري",
        "patient_id": "PID-2026-106",
        "age": 38, "gender": 1,
        "doc_idx": 2,
        "dept": "Gastroenterology",
        "profile": {
            'age': 38, 'gender': 1, 'bmi': 23.9, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 1.4, 'albumin': 3.9, 'alp': 98, 'alt': 58, 'ast': 52,
            'platelets': 235000, 'prothrombin': 12.8, 'copper': 118, 'cholesterol': 178,
            'creatinine': 0.9, 'glucose': 92, 'ggt': 60, 'triglycerides': 115,
            'uric_acid': 5.2, 'hdl': 50, 'total_proteins': 7.3
        }
    },
    {
        "name": "سامي كمال الزهراني",
        "patient_id": "PID-2026-107",
        "age": 64, "gender": 1,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 64, 'gender': 1, 'bmi': 25.5, 'smoking': 'Yes', 'alcohol': 'Moderate',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'Medium',
            'ascites': 'Yes', 'hepatomegaly': 'Yes', 'spiders': 'Yes', 'edema': 'Severe',
            'bilirubin': 4.1, 'albumin': 2.7, 'alp': 195, 'alt': 145, 'ast': 175,
            'platelets': 105000, 'prothrombin': 18.5, 'copper': 185, 'cholesterol': 145,
            'creatinine': 1.4, 'glucose': 135, 'ggt': 210, 'triglycerides': 155,
            'uric_acid': 7.6, 'hdl': 34, 'total_proteins': 6.2
        }
    },

    # --- FATTY LIVER (6 Cases) ---
    {
        "name": "فاطمة إبراهيم العلي",
        "patient_id": "PID-2026-108",
        "age": 46, "gender": 0,
        "doc_idx": 1,
        "dept": "Oncology",
        "profile": {
            'age': 46, 'gender': 0, 'bmi': 31.5, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'Medium',
            'ascites': 'No', 'hepatomegaly': 'Yes', 'spiders': 'No', 'edema': 'Slight',
            'bilirubin': 1.5, 'albumin': 3.9, 'alp': 115, 'alt': 82, 'ast': 64,
            'platelets': 240000, 'prothrombin': 13.2, 'copper': 125, 'cholesterol': 245,
            'creatinine': 0.9, 'glucose': 118, 'ggt': 85, 'triglycerides': 235,
            'uric_acid': 6.4, 'hdl': 40, 'total_proteins': 7.4
        }
    },
    {
        "name": "نورة سلطان المهندي",
        "patient_id": "PID-2026-109",
        "age": 52, "gender": 0,
        "doc_idx": 2,
        "dept": "Gastroenterology",
        "profile": {
            'age': 52, 'gender': 0, 'bmi': 33.2, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'Medium',
            'ascites': 'No', 'hepatomegaly': 'Yes', 'spiders': 'No', 'edema': 'Moderate',
            'bilirubin': 1.7, 'albumin': 3.7, 'alp': 125, 'alt': 94, 'ast': 72,
            'platelets': 225000, 'prothrombin': 13.8, 'copper': 130, 'cholesterol': 260,
            'creatinine': 1.0, 'glucose': 132, 'ggt': 98, 'triglycerides': 265,
            'uric_acid': 6.8, 'hdl': 36, 'total_proteins': 7.2
        }
    },
    {
        "name": "هند راشد الكعبي",
        "patient_id": "PID-2026-110",
        "age": 39, "gender": 0,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 39, 'gender': 0, 'bmi': 29.4, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Moderate', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 1.2, 'albumin': 4.1, 'alp': 98, 'alt': 55, 'ast': 45,
            'platelets': 260000, 'prothrombin': 12.8, 'copper': 115, 'cholesterol': 220,
            'creatinine': 0.8, 'glucose': 102, 'ggt': 58, 'triglycerides': 195,
            'uric_acid': 5.6, 'hdl': 46, 'total_proteins': 7.6
        }
    },
    {
        "name": "زياد بدر الصباح",
        "patient_id": "PID-2026-111",
        "age": 48, "gender": 1,
        "doc_idx": 1,
        "dept": "Oncology",
        "profile": {
            'age': 48, 'gender': 1, 'bmi': 32.0, 'smoking': 'Yes', 'alcohol': 'Moderate',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'Medium',
            'ascites': 'No', 'hepatomegaly': 'Yes', 'spiders': 'No', 'edema': 'Slight',
            'bilirubin': 1.6, 'albumin': 3.8, 'alp': 120, 'alt': 89, 'ast': 68,
            'platelets': 230000, 'prothrombin': 13.5, 'copper': 128, 'cholesterol': 255,
            'creatinine': 1.0, 'glucose': 124, 'ggt': 92, 'triglycerides': 250,
            'uric_acid': 6.6, 'hdl': 38, 'total_proteins': 7.3
        }
    },
    {
        "name": "منى صالح البلوشي",
        "patient_id": "PID-2026-112",
        "age": 55, "gender": 0,
        "doc_idx": 2,
        "dept": "Gastroenterology",
        "profile": {
            'age': 55, 'gender': 0, 'bmi': 30.8, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Low', 'cancer_history': 'No', 'genetic_risk': 'Medium',
            'ascites': 'No', 'hepatomegaly': 'Yes', 'spiders': 'No', 'edema': 'Slight',
            'bilirubin': 1.4, 'albumin': 3.9, 'alp': 112, 'alt': 76, 'ast': 58,
            'platelets': 245000, 'prothrombin': 13.0, 'copper': 122, 'cholesterol': 238,
            'creatinine': 0.9, 'glucose': 114, 'ggt': 78, 'triglycerides': 225,
            'uric_acid': 6.2, 'hdl': 42, 'total_proteins': 7.5
        }
    },
    {
        "name": "سحر ماجد الرويلي",
        "patient_id": "PID-2026-113",
        "age": 44, "gender": 0,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 44, 'gender': 0, 'bmi': 29.8, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Moderate', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 1.3, 'albumin': 4.0, 'alp': 102, 'alt': 62, 'ast': 48,
            'platelets': 255000, 'prothrombin': 12.9, 'copper': 118, 'cholesterol': 228,
            'creatinine': 0.8, 'glucose': 106, 'ggt': 64, 'triglycerides': 205,
            'uric_acid': 5.8, 'hdl': 45, 'total_proteins': 7.6
        }
    },

    # --- TUMOR / CANCER RISK (4 Cases) ---
    {
        "name": "خالد فهد العتيبي",
        "patient_id": "PID-2026-114",
        "age": 67, "gender": 1,
        "doc_idx": 1,
        "dept": "Oncology",
        "profile": {
            'age': 67, 'gender': 1, 'bmi': 26.5, 'smoking': 'Yes', 'alcohol': 'Heavy',
            'activity': 'Low', 'cancer_history': 'Yes', 'genetic_risk': 'High',
            'ascites': 'Slight', 'hepatomegaly': 'Yes', 'spiders': 'Yes', 'edema': 'Moderate',
            'bilirubin': 2.8, 'albumin': 3.1, 'alp': 185, 'alt': 92, 'ast': 115,
            'platelets': 140000, 'prothrombin': 16.5, 'copper': 165, 'cholesterol': 175,
            'creatinine': 1.3, 'glucose': 122, 'ggt': 160, 'triglycerides': 165,
            'uric_acid': 7.1, 'hdl': 37, 'total_proteins': 6.5
        }
    },
    {
        "name": "لولوة جاسم الهاجري",
        "patient_id": "PID-2026-115",
        "age": 62, "gender": 0,
        "doc_idx": 2,
        "dept": "Gastroenterology",
        "profile": {
            'age': 62, 'gender': 0, 'bmi': 27.8, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Low', 'cancer_history': 'Yes', 'genetic_risk': 'High',
            'ascites': 'No', 'hepatomegaly': 'Yes', 'spiders': 'No', 'edema': 'Slight',
            'bilirubin': 2.3, 'albumin': 3.3, 'alp': 165, 'alt': 78, 'ast': 88,
            'platelets': 175000, 'prothrombin': 15.0, 'copper': 155, 'cholesterol': 190,
            'creatinine': 1.1, 'glucose': 115, 'ggt': 135, 'triglycerides': 175,
            'uric_acid': 6.7, 'hdl': 39, 'total_proteins': 6.9
        }
    },
    {
        "name": "عمر عبد العزيز الحربي",
        "patient_id": "PID-2026-116",
        "age": 71, "gender": 1,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 71, 'gender': 1, 'bmi': 25.2, 'smoking': 'Yes', 'alcohol': 'Moderate',
            'activity': 'Low', 'cancer_history': 'Yes', 'genetic_risk': 'High',
            'ascites': 'Yes', 'hepatomegaly': 'Yes', 'spiders': 'Yes', 'edema': 'Severe',
            'bilirubin': 3.6, 'albumin': 2.8, 'alp': 220, 'alt': 105, 'ast': 138,
            'platelets': 115000, 'prothrombin': 17.8, 'copper': 180, 'cholesterol': 150,
            'creatinine': 1.4, 'glucose': 138, 'ggt': 195, 'triglycerides': 160,
            'uric_acid': 7.5, 'hdl': 33, 'total_proteins': 6.3
        }
    },
    {
        "name": "فهد يحيى العنزي",
        "patient_id": "PID-2026-117",
        "age": 59, "gender": 1,
        "doc_idx": 1,
        "dept": "Oncology",
        "profile": {
            'age': 59, 'gender': 1, 'bmi': 28.0, 'smoking': 'Yes', 'alcohol': 'Heavy',
            'activity': 'Low', 'cancer_history': 'Yes', 'genetic_risk': 'Medium',
            'ascites': 'No', 'hepatomegaly': 'Yes', 'spiders': 'No', 'edema': 'Slight',
            'bilirubin': 2.0, 'albumin': 3.5, 'alp': 150, 'alt': 70, 'ast': 82,
            'platelets': 185000, 'prothrombin': 14.6, 'copper': 145, 'cholesterol': 185,
            'creatinine': 1.1, 'glucose': 118, 'ggt': 125, 'triglycerides': 170,
            'uric_acid': 6.5, 'hdl': 41, 'total_proteins': 7.0
        }
    },

    # --- HEALTHY / NORMAL LIVER (8 Cases) ---
    {
        "name": "محمد عبد الرحمن الدوسري",
        "patient_id": "PID-2026-118",
        "age": 31, "gender": 1,
        "doc_idx": 2,
        "dept": "Gastroenterology",
        "profile": {
            'age': 31, 'gender': 1, 'bmi': 22.8, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.7, 'albumin': 4.4, 'alp': 72, 'alt': 22, 'ast': 24,
            'platelets': 285000, 'prothrombin': 12.0, 'copper': 105, 'cholesterol': 170,
            'creatinine': 0.9, 'glucose': 88, 'ggt': 22, 'triglycerides': 110,
            'uric_acid': 5.1, 'hdl': 55, 'total_proteins': 7.6
        }
    },
    {
        "name": "ريم عادل الغامدي",
        "patient_id": "PID-2026-119",
        "age": 27, "gender": 0,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 27, 'gender': 0, 'bmi': 21.5, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'High', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.6, 'albumin': 4.6, 'alp': 68, 'alt': 18, 'ast': 20,
            'platelets': 310000, 'prothrombin': 11.8, 'copper': 98, 'cholesterol': 165,
            'creatinine': 0.7, 'glucose': 84, 'ggt': 18, 'triglycerides': 95,
            'uric_acid': 4.5, 'hdl': 62, 'total_proteins': 7.8
        }
    },
    {
        "name": "دانة فيصل الخالدي",
        "patient_id": "PID-2026-120",
        "age": 34, "gender": 0,
        "doc_idx": 1,
        "dept": "Oncology",
        "profile": {
            'age': 34, 'gender': 0, 'bmi': 23.2, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.8, 'albumin': 4.3, 'alp': 75, 'alt': 24, 'ast': 26,
            'platelets': 275000, 'prothrombin': 12.2, 'copper': 110, 'cholesterol': 175,
            'creatinine': 0.8, 'glucose': 90, 'ggt': 25, 'triglycerides': 115,
            'uric_acid': 4.9, 'hdl': 54, 'total_proteins': 7.5
        }
    },
    {
        "name": "ماجد وليد السليمان",
        "patient_id": "PID-2026-121",
        "age": 36, "gender": 1,
        "doc_idx": 2,
        "dept": "Gastroenterology",
        "profile": {
            'age': 36, 'gender': 1, 'bmi': 24.1, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.9, 'albumin': 4.5, 'alp': 78, 'alt': 26, 'ast': 27,
            'platelets': 290000, 'prothrombin': 12.1, 'copper': 112, 'cholesterol': 180,
            'creatinine': 0.9, 'glucose': 91, 'ggt': 28, 'triglycerides': 120,
            'uric_acid': 5.3, 'hdl': 52, 'total_proteins': 7.7
        }
    },
    {
        "name": "حنان إسماعيل النجار",
        "patient_id": "PID-2026-122",
        "age": 29, "gender": 0,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 29, 'gender': 0, 'bmi': 22.0, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.6, 'albumin': 4.5, 'alp': 65, 'alt': 19, 'ast': 21,
            'platelets': 295000, 'prothrombin': 11.9, 'copper': 102, 'cholesterol': 168,
            'creatinine': 0.7, 'glucose': 86, 'ggt': 20, 'triglycerides': 98,
            'uric_acid': 4.6, 'hdl': 58, 'total_proteins': 7.6
        }
    },
    {
        "name": "هيثم عصام الشريف",
        "patient_id": "PID-2026-123",
        "age": 41, "gender": 1,
        "doc_idx": 1,
        "dept": "Oncology",
        "profile": {
            'age': 41, 'gender': 1, 'bmi': 23.5, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Moderate', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.8, 'albumin': 4.4, 'alp': 80, 'alt': 28, 'ast': 29,
            'platelets': 270000, 'prothrombin': 12.3, 'copper': 115, 'cholesterol': 182,
            'creatinine': 1.0, 'glucose': 93, 'ggt': 32, 'triglycerides': 125,
            'uric_acid': 5.4, 'hdl': 50, 'total_proteins': 7.5
        }
    },
    {
        "name": "أسماء إبراهيم السعد",
        "patient_id": "PID-2026-124",
        "age": 33, "gender": 0,
        "doc_idx": 2,
        "dept": "Gastroenterology",
        "profile": {
            'age': 33, 'gender': 0, 'bmi': 22.4, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.7, 'albumin': 4.6, 'alp': 70, 'alt': 21, 'ast': 23,
            'platelets': 305000, 'prothrombin': 12.0, 'copper': 100, 'cholesterol': 162,
            'creatinine': 0.7, 'glucose': 87, 'ggt': 21, 'triglycerides': 105,
            'uric_acid': 4.7, 'hdl': 60, 'total_proteins': 7.7
        }
    },
    {
        "name": "بشير طلال الجابري",
        "patient_id": "PID-2026-125",
        "age": 45, "gender": 1,
        "doc_idx": 0,
        "dept": "Hepatology",
        "profile": {
            'age': 45, 'gender': 1, 'bmi': 24.8, 'smoking': 'No', 'alcohol': 'No',
            'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
            'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
            'bilirubin': 0.9, 'albumin': 4.3, 'alp': 82, 'alt': 29, 'ast': 30,
            'platelets': 265000, 'prothrombin': 12.4, 'copper': 118, 'cholesterol': 185,
            'creatinine': 1.0, 'glucose': 94, 'ggt': 35, 'triglycerides': 130,
            'uric_acid': 5.5, 'hdl': 49, 'total_proteins': 7.4
        }
    }
]

DOCTORS = [
    {
        "username": "dr_sarah",
        "email": "dr.sarah@hepatiq.com",
        "password": "Doctor@2026",
        "full_name": "د. سارة المنصوري",
        "role": "doctor"
    },
    {
        "username": "dr_omar",
        "email": "dr.omar@hepatiq.com",
        "password": "Doctor@2026",
        "full_name": "د. عمر الفاروق",
        "role": "doctor"
    },
    {
        "username": "dr_layla",
        "email": "dr.layla@hepatiq.com",
        "password": "Doctor@2026",
        "full_name": "د. ليلى الهاشمي",
        "role": "doctor"
    }
]

def main():
    print("Connecting to live site https://www.hepatiq.site...")
    admin_session = requests.Session()
    login_res = admin_session.post(f"{BASE_URL}/auth/login", json={
        "username": "verify_admin",
        "password": "VerifyPass123!"
    })

    if login_res.status_code != 200:
        print(f"Failed to log in as admin: {login_res.status_code} {login_res.text}")
        return

    print("Admin logged in successfully!")

    print("\n--- 1. Registering Doctors ---")
    for doc in DOCTORS:
        reg_res = admin_session.post(f"{BASE_URL}/auth/register", json=doc)
        if reg_res.status_code == 200:
            print(f"Registered doctor: {doc['full_name']} (@{doc['username']})")
        elif "already exists" in reg_res.text:
            print(f"Doctor @{doc['username']} already registered.")
        else:
            print(f"Note on doctor @{doc['username']}: {reg_res.status_code} {reg_res.text}")

    # Fetch existing live patients
    p_list_res = admin_session.get(f"{BASE_URL}/patients")
    existing_pids = set()
    if p_list_res.status_code == 200:
        for p in p_list_res.json().get("patients", []):
            existing_pids.add(p.get("patient_id"))
    print(f"\nExisting patients on live server: {len(existing_pids)}")

    # Sessions for doctors
    doc_sessions = []
    for doc in DOCTORS:
        s = requests.Session()
        l_res = s.post(f"{BASE_URL}/auth/login", json={"username": doc["username"], "password": doc["password"]})
        if l_res.status_code == 200:
            doc_sessions.append(s)
        else:
            print(f"Warning: could not log in as {doc['username']}, using admin session fallback")
            doc_sessions.append(admin_session)

    print("\n--- 2. Creating Patients & Running AI Analysis ---")
    for idx, p_def in enumerate(PATIENTS_DATA):
        doc_s = doc_sessions[p_def["doc_idx"]]
        doc_info = DOCTORS[p_def["doc_idx"]]

        pid = p_def["patient_id"]
        # Calculate birth date
        birth_year = 2026 - p_def["age"]
        birth_date = f"{birth_year}-0{random.randint(1,9)}-{random.randint(10,28)}"

        patient_body = {
            "name": p_def["name"],
            "patient_id": pid,
            "birth_date": birth_date,
            "email": f"{pid.lower()}@patient.hepatiq.com",
            "phone": f"+9665{random.randint(10000000, 99999999)}",
            "department": p_def["dept"],
            "doctor_name": doc_info["full_name"]
        }

        # Create patient
        p_res = doc_s.post(f"{BASE_URL}/patients", json=patient_body)
        if p_res.status_code == 200:
            p_data = p_res.json().get("patient", {})
            p_db_id = p_data.get("id")
            print(f"[{idx+1}/25] Created Patient {p_def['name']} (ID: {p_db_id}) -> Assigned to {doc_info['full_name']}")
        elif "Currently used" in p_res.text:
            print(f"[{idx+1}/25] Patient {pid} already exists on live site.")
            # Find its db ID
            p_find = admin_session.get(f"{BASE_URL}/patients?patient_id={pid}")
            p_db_id = None
            if p_find.status_code == 200:
                pts = p_find.json().get("patients", [])
                for p in pts:
                    if p.get("patient_id") == pid:
                        p_db_id = p.get("id")
                        break
            if not p_db_id:
                continue
        else:
            print(f"Error creating patient {pid}: {p_res.status_code} {p_res.text}")
            continue

        # Run AI analysis for patient
        profile_body = dict(p_def["profile"])
        profile_body["patient_id"] = p_db_id

        ana_res = doc_s.post(f"{BASE_URL}/analyze", json=profile_body)
        if ana_res.status_code == 200:
            ana_data = ana_res.json()
            print(f"    -> AI Diagnosis: {ana_data.get('diagnosis')} | Risk: {ana_data.get('risk_level')}")
        else:
            print(f"    -> Analysis error: {ana_res.status_code} {ana_res.text[:100]}")

        time.sleep(0.3)

    print("\n=== VERIFICATION ON LIVE SITE ===")
    final_p = admin_session.get(f"{BASE_URL}/patients")
    if final_p.status_code == 200:
        patients_count = len(final_p.json().get("patients", []))
        print(f"TOTAL PATIENTS ON LIVE SITE NOW: {patients_count}")

if __name__ == "__main__":
    main()
