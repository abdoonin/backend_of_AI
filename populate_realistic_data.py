#!/usr/bin/env python3
"""
Populate database with 3 realistic doctor accounts and 25 diverse clinical patient cases.
All cases run through the clinical DiagnosisEngine or realistic ML structures,
generating authentic lab tests, medical reports, APRI scores, and detailed findings.
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta, timezone

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal, engine, Base
from models import Patient, LabTest, MedicalReport, User, DOCTOR_PRESET_PERMISSIONS
from auth import hash_password
from diagnosis_engine import DiagnosisEngine

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, (int, float, bool)):
        return obj
    elif hasattr(obj, 'tolist'):
        return make_serializable(obj.tolist())
    return obj

def populate():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    print("=== 1. Creating Doctor Accounts ===")
    doctors_info = [
        {
            "username": "dr_sarah",
            "email": "dr.sarah@hepatiq.com",
            "full_name": "د. سارة المنصوري",
            "role": "doctor",
            "password": "Doctor@2026",
            "dept": "Hepatology"
        },
        {
            "username": "dr_omar",
            "email": "dr.omar@hepatiq.com",
            "full_name": "د. عمر الفاروق",
            "role": "doctor",
            "password": "Doctor@2026",
            "dept": "Oncology"
        },
        {
            "username": "dr_layla",
            "email": "dr.layla@hepatiq.com",
            "full_name": "د. ليلى الهاشمي",
            "role": "doctor",
            "password": "Doctor@2026",
            "dept": "Gastroenterology"
        }
    ]

    doctor_objs = []
    for doc in doctors_info:
        user = db.query(User).filter(User.username == doc["username"]).first()
        if not user:
            user = User(
                username=doc["username"],
                email=doc["email"],
                full_name=doc["full_name"],
                role=doc["role"],
                hashed_password=hash_password(doc["password"]),
                is_active=1,
                permissions=json.dumps(DOCTOR_PRESET_PERMISSIONS),
                created_at=datetime.now(timezone.utc) - timedelta(days=120)
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created doctor: {doc['full_name']} (@{doc['username']})")
        else:
            print(f"Doctor @{doc['username']} already exists.")
        doctor_objs.append(user)

    print("\n=== 2. Generating 25 Diverse Clinical Patient Cases ===")
    
    # 25 Patient Profiles
    # Clinical Categories:
    # 1. Hepatitis C (7 cases)
    # 2. Fatty Liver (6 cases)
    # 3. High Tumor / Cancer Risk (4 cases)
    # 4. Healthy / Normal Liver (8 cases)

    patients_definitions = [
        # --- HEPATITIS C (7 Cases) ---
        {
            "name": "أحمد خالد المنصور",
            "patient_id": "PID-2026-101",
            "age": 54, "gender": "Male",
            "category": "hepatitis",
            "profile": {
                'age': 54, 'gender': 'Male', 'bmi': 27.2, 'smoking': 'Yes', 'alcohol': 'Moderate',
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
            "age": 49, "gender": "Female",
            "category": "hepatitis",
            "profile": {
                'age': 49, 'gender': 'Female', 'bmi': 25.1, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 61, "gender": "Male",
            "category": "hepatitis",
            "profile": {
                'age': 61, 'gender': 'Male', 'bmi': 26.8, 'smoking': 'Yes', 'alcohol': 'Heavy',
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
            "age": 43, "gender": "Female",
            "category": "hepatitis",
            "profile": {
                'age': 43, 'gender': 'Female', 'bmi': 24.3, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 57, "gender": "Male",
            "category": "hepatitis",
            "profile": {
                'age': 57, 'gender': 'Male', 'bmi': 28.4, 'smoking': 'Yes', 'alcohol': 'Moderate',
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
            "age": 38, "gender": "Male",
            "category": "hepatitis",
            "profile": {
                'age': 38, 'gender': 'Male', 'bmi': 23.9, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 64, "gender": "Male",
            "category": "hepatitis",
            "profile": {
                'age': 64, 'gender': 'Male', 'bmi': 25.5, 'smoking': 'Yes', 'alcohol': 'Moderate',
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
            "age": 46, "gender": "Female",
            "category": "fatty_liver",
            "profile": {
                'age': 46, 'gender': 'Female', 'bmi': 31.5, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 52, "gender": "Female",
            "category": "fatty_liver",
            "profile": {
                'age': 52, 'gender': 'Female', 'bmi': 33.2, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 39, "gender": "Female",
            "category": "fatty_liver",
            "profile": {
                'age': 39, 'gender': 'Female', 'bmi': 29.4, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 48, "gender": "Male",
            "category": "fatty_liver",
            "profile": {
                'age': 48, 'gender': 'Male', 'bmi': 32.0, 'smoking': 'Yes', 'alcohol': 'Moderate',
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
            "age": 55, "gender": "Female",
            "category": "fatty_liver",
            "profile": {
                'age': 55, 'gender': 'Female', 'bmi': 30.8, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 44, "gender": "Female",
            "category": "fatty_liver",
            "profile": {
                'age': 44, 'gender': 'Female', 'bmi': 29.8, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 67, "gender": "Male",
            "category": "cancer",
            "profile": {
                'age': 67, 'gender': 'Male', 'bmi': 26.5, 'smoking': 'Yes', 'alcohol': 'Heavy',
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
            "age": 62, "gender": "Female",
            "category": "cancer",
            "profile": {
                'age': 62, 'gender': 'Female', 'bmi': 27.8, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 71, "gender": "Male",
            "category": "cancer",
            "profile": {
                'age': 71, 'gender': 'Male', 'bmi': 25.2, 'smoking': 'Yes', 'alcohol': 'Moderate',
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
            "age": 59, "gender": "Male",
            "category": "cancer",
            "profile": {
                'age': 59, 'gender': 'Male', 'bmi': 28.0, 'smoking': 'Yes', 'alcohol': 'Heavy',
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
            "age": 31, "gender": "Male",
            "category": "healthy",
            "profile": {
                'age': 31, 'gender': 'Male', 'bmi': 22.8, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 27, "gender": "Female",
            "category": "healthy",
            "profile": {
                'age': 27, 'gender': 'Female', 'bmi': 21.5, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 34, "gender": "Female",
            "category": "healthy",
            "profile": {
                'age': 34, 'gender': 'Female', 'bmi': 23.2, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 36, "gender": "Male",
            "category": "healthy",
            "profile": {
                'age': 36, 'gender': 'Male', 'bmi': 24.1, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 29, "gender": "Female",
            "category": "healthy",
            "profile": {
                'age': 29, 'gender': 'Female', 'bmi': 22.0, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 41, "gender": "Male",
            "category": "healthy",
            "profile": {
                'age': 41, 'gender': 'Male', 'bmi': 23.5, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 33, "gender": "Female",
            "category": "healthy",
            "profile": {
                'age': 33, 'gender': 'Female', 'bmi': 22.4, 'smoking': 'No', 'alcohol': 'No',
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
            "age": 45, "gender": "Male",
            "category": "healthy",
            "profile": {
                'age': 45, 'gender': 'Male', 'bmi': 24.8, 'smoking': 'No', 'alcohol': 'No',
                'activity': 'Regular', 'cancer_history': 'No', 'genetic_risk': 'Low',
                'ascites': 'No', 'hepatomegaly': 'No', 'spiders': 'No', 'edema': 'No',
                'bilirubin': 0.9, 'albumin': 4.3, 'alp': 82, 'alt': 29, 'ast': 30,
                'platelets': 265000, 'prothrombin': 12.4, 'copper': 118, 'cholesterol': 185,
                'creatinine': 1.0, 'glucose': 94, 'ggt': 35, 'triglycerides': 130,
                'uric_acid': 5.5, 'hdl': 49, 'total_proteins': 7.4
            }
        }
    ]

    engine_instance = DiagnosisEngine()

    departments = ["Hepatology", "Oncology", "Gastroenterology", "Internal Medicine"]

    # Assign each patient to a doctor and create records
    for idx, p_def in enumerate(patients_definitions):
        assigned_doc = doctor_objs[idx % len(doctor_objs)]
        dept = assigned_doc.full_name.split()[0] # fallback
        if "سارة" in assigned_doc.full_name:
            dept = "Hepatology"
        elif "عمر" in assigned_doc.full_name:
            dept = "Oncology"
        else:
            dept = "Gastroenterology"

        existing_patient = db.query(Patient).filter(Patient.patient_id == p_def["patient_id"]).first()
        if existing_patient:
            print(f"Skipping existing patient {p_def['patient_id']}")
            continue

        # Calculate birth date from age
        birth_year = 2026 - p_def["age"]
        birth_date = f"{birth_year}-0{random.randint(1,9)}-{random.randint(10,28)}"

        patient = Patient(
            name=p_def["name"],
            patient_id=p_def["patient_id"],
            birth_date=birth_date,
            email=f"{p_def['patient_id'].lower()}@patient.hepatiq.com",
            phone=f"+9665{random.randint(10000000, 99999999)}",
            department=dept,
            doctor_name=assigned_doc.full_name,
            doctor_id=assigned_doc.id,
            status="active",
            created_at=datetime.now(timezone.utc) - timedelta(days=random.randint(5, 60))
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

        # Run DiagnosisEngine
        prof = p_def["profile"]
        try:
            raw_result = engine_instance.predict_full_diagnosis(prof)
            clean_res = make_serializable(raw_result)
        except Exception as e:
            print(f"Error evaluating diagnosis for {patient.name}: {e}")
            continue

        diagnosis = clean_res.get('diagnosis', 'Normal Liver Function')
        confidence = float(clean_res.get('confidence', 95.0))
        advice = clean_res.get('advice', 'Routine follow up advised.')
        risk_level = clean_res.get('risk_level', 'low')
        risk_map = {'low': 'low', 'moderate': 'medium', 'high': 'high'}
        db_risk = risk_map.get(risk_level, 'low')

        det_results = clean_res.get('detailed_results', {})
        # Enrich det_results with inputs so the UI can display inputs, gender, age
        inputs_blob = {k: str(v) for k, v in prof.items()}
        inputs_blob['patient_name'] = patient.name
        inputs_blob['patient_id'] = patient.patient_id
        inputs_blob['doctor_name'] = assigned_doc.full_name
        inputs_blob['department'] = dept
        det_results['inputs'] = inputs_blob

        report = MedicalReport(
            patient_id=patient.id,
            doctor_id=assigned_doc.id,
            diagnosis=diagnosis,
            confidence=confidence,
            advice=advice,
            risk_level=db_risk,
            is_finalized=1,
            detailed_results=json.dumps(det_results),
            created_at=patient.created_at + timedelta(hours=2)
        )
        db.add(report)

        # Create realistic Lab Tests
        lab_specs = [
            ("ALT", prof['alt'], "U/L", "7 - 56", "high" if prof['alt'] > 56 else "normal"),
            ("AST", prof['ast'], "U/L", "10 - 40", "high" if prof['ast'] > 40 else "normal"),
            ("Total Bilirubin", prof['bilirubin'], "mg/dL", "0.2 - 1.2", "high" if prof['bilirubin'] > 1.2 else "normal"),
            ("Albumin", prof['albumin'], "g/dL", "3.5 - 5.0", "low" if prof['albumin'] < 3.5 else "normal"),
            ("Alkaline Phosphatase (ALP)", prof['alp'], "U/L", "44 - 147", "high" if prof['alp'] > 147 else "normal"),
            ("Platelets", prof['platelets'], "x10^3/uL", "150,000 - 450,000", "low" if prof['platelets'] < 150000 else "normal"),
            ("GGT", prof['ggt'], "U/L", "9 - 48", "high" if prof['ggt'] > 48 else "normal"),
            ("Total Cholesterol", prof['cholesterol'], "mg/dL", "< 200", "high" if prof['cholesterol'] > 200 else "normal"),
            ("Triglycerides", prof['triglycerides'], "mg/dL", "< 150", "high" if prof['triglycerides'] > 150 else "normal"),
            ("Serum Creatinine", prof['creatinine'], "mg/dL", "0.6 - 1.2", "high" if prof['creatinine'] > 1.2 else "normal"),
            ("Fasting Glucose", prof['glucose'], "mg/dL", "70 - 99", "high" if prof['glucose'] > 99 else "normal")
        ]

        for t_name, val, unit, n_range, st in lab_specs:
            lab_test = LabTest(
                patient_id=patient.id,
                test_name=t_name,
                value=float(val),
                unit=unit,
                normal_range=n_range,
                status=st,
                date=patient.created_at + timedelta(hours=1),
                created_at=patient.created_at + timedelta(hours=1)
            )
            db.add(lab_test)

        db.commit()
        print(f"[{idx+1}/25] Added Patient {patient.name} ({p_def['category']}) -> Assigned to {assigned_doc.full_name}")

    db.close()
    print("\nSUCCESS: All 3 Doctors and 25 Patients have been populated into the database!")

if __name__ == "__main__":
    populate()
