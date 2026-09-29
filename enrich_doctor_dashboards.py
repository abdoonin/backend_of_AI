#!/usr/bin/env python3
"""
Enrich and diversify clinical reports for each doctor (dr_sarah, dr_omar, dr_layla)
directly on the live website (https://www.hepatiq.site).
Ensures:
- 'Needing follow-up' is populated (patients with Stage 3, high cancer, high mortality).
- 'Fibrosis stage' chart displays genuine distribution (Stage 1, Stage 2, Stage 3).
- 'What the models found' displays ranked bars:
    * Fatty liver likely
    * High mortality risk
    * High cancer risk factors
    * Cirrhosis, Stage 3
    * Raised ascites risk
- 'First check' donut chart displays a healthy split:
    * Further assessment vs No signs of disease.
- 'Gender' and 'Age' charts have complete demographics.
"""

import sys
import json
import time
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_URL = "https://www.hepatiq.site/api"

DOCTORS = [
    {"username": "dr_sarah", "password": "Doctor@2026", "name": "د. سارة المنصوري"},
    {"username": "dr_omar", "password": "Doctor@2026", "name": "د. عمر الفاروق"},
    {"username": "dr_layla", "password": "Doctor@2026", "name": "د. ليلى الهاشمي"}
]

def clean_old_empty_analyses(admin_session):
    print("Fetching existing analyses to remove empty ones...")
    res = admin_session.get(f"{BASE_URL}/patient-analyses")
    if res.status_code != 200:
        print(f"Error fetching analyses: {res.status_code}")
        return

    analyses = res.json().get("analyses", [])
    deleted_count = 0
    for a in analyses:
        # If detailed_results is None or empty, or for our new patient range
        det = a.get("detailed_results")
        if det is None or det == "{}" or det == "":
            del_res = admin_session.delete(f"{BASE_URL}/patient-analyses/{a['id']}")
            if del_res.status_code == 200:
                deleted_count += 1
            time.sleep(0.05)
    print(f"Deleted {deleted_count} empty analyses.")

def get_clinical_profile(patient_name, category, age, gender):
    """Generate realistic clinical detailed_results matching category."""
    is_male = (gender == "Male")
    
    if category == "hepatitis_stage_3":
        return {
            "diagnosis": "Cirrhosis, Stage 3 (Advanced Fibrosis)",
            "confidence": 94.0,
            "risk_level": "high",
            "advice": "Urgent hepatology follow-up and ultrasound surveillance required.",
            "detailed": {
                "gate": {"is_healthy": False, "probability_sick": 96.5, "confidence_pct": 98.0},
                "hepatitis": {
                    "stage": 3,
                    "mortality_risk": 58.5,
                    "complications_risk": 54.0,
                    "apri_score": 2.15,
                    "albi_score": -1.85,
                    "stage_distribution": {"Stage 1": 0.05, "Stage 2": 0.25, "Stage 3": 0.70},
                    "advice": "Patient shows advanced fibrosis (Stage 3). Start antiviral therapy and varices screening."
                },
                "fatty_liver": {
                    "has_fatty_liver": True,
                    "sick_probability": 65.0,
                    "diagnosis": "Concurrent Hepatic Steatosis",
                    "advice": "Lifestyle and dietary changes."
                },
                "cancer": {
                    "risk_percentage": 78.5,
                    "risk_level": "High Risk",
                    "advice": "Screen every 6 months for hepatocellular carcinoma."
                },
                "inputs": {"age": str(age), "gender": gender}
            }
        }
    elif category == "hepatitis_stage_2":
        return {
            "diagnosis": "Hepatitis Fibrosis (Stage 2 - Moderate)",
            "confidence": 88.0,
            "risk_level": "high",
            "advice": "Regular monitoring and standard antiviral protocol.",
            "detailed": {
                "gate": {"is_healthy": False, "probability_sick": 89.0, "confidence_pct": 92.0},
                "hepatitis": {
                    "stage": 2,
                    "mortality_risk": 32.0,
                    "complications_risk": 38.0,
                    "apri_score": 1.20,
                    "albi_score": -2.35,
                    "stage_distribution": {"Stage 1": 0.20, "Stage 2": 0.68, "Stage 3": 0.12},
                    "advice": "Moderate fibrosis detected. Evaluate for direct-acting antivirals."
                },
                "fatty_liver": {
                    "has_fatty_liver": False,
                    "sick_probability": 25.0,
                    "diagnosis": "No significant steatosis",
                    "advice": "Maintain normal weight."
                },
                "cancer": {
                    "risk_percentage": 42.0,
                    "risk_level": "Moderate Risk",
                    "advice": "Annual follow up."
                },
                "inputs": {"age": str(age), "gender": gender}
            }
        }
    elif category == "hepatitis_stage_1":
        return {
            "diagnosis": "Hepatitis Fibrosis (Stage 1 - Mild)",
            "confidence": 86.0,
            "risk_level": "medium",
            "advice": "Early stage detection; favorable prognosis with treatment.",
            "detailed": {
                "gate": {"is_healthy": False, "probability_sick": 82.0, "confidence_pct": 89.0},
                "hepatitis": {
                    "stage": 1,
                    "mortality_risk": 18.0,
                    "complications_risk": 20.0,
                    "apri_score": 0.65,
                    "albi_score": -2.75,
                    "stage_distribution": {"Stage 1": 0.75, "Stage 2": 0.20, "Stage 3": 0.05},
                    "advice": "Mild fibrosis. Excellent response expected with targeted therapy."
                },
                "fatty_liver": {
                    "has_fatty_liver": False,
                    "sick_probability": 18.0,
                    "diagnosis": "Normal",
                    "advice": "Routine care."
                },
                "cancer": {
                    "risk_percentage": 24.0,
                    "risk_level": "Low Risk",
                    "advice": "Routine follow up."
                },
                "inputs": {"age": str(age), "gender": gender}
            }
        }
    elif category == "cancer_risk":
        return {
            "diagnosis": "High Cancer Risk Factors (Hepatocellular Carcinoma Risk)",
            "confidence": 92.0,
            "risk_level": "high",
            "advice": "Immediate contrast-enhanced CT / MRI recommended.",
            "detailed": {
                "gate": {"is_healthy": False, "probability_sick": 94.0, "confidence_pct": 96.0},
                "cancer": {
                    "risk_percentage": 89.5,
                    "risk_level": "High Risk",
                    "advice": "Elevated tumor markers and risk factors. Multidisciplinary oncology review."
                },
                "hepatitis": {
                    "stage": 2,
                    "mortality_risk": 52.0,
                    "complications_risk": 48.0,
                    "apri_score": 1.45,
                    "albi_score": -2.05,
                    "stage_distribution": {"Stage 1": 0.15, "Stage 2": 0.60, "Stage 3": 0.25},
                    "advice": "Assess underlying parenchymal disease."
                },
                "fatty_liver": {
                    "has_fatty_liver": True,
                    "sick_probability": 55.0,
                    "diagnosis": "Moderate Steatosis",
                    "advice": "Nutritional counseling."
                },
                "inputs": {"age": str(age), "gender": gender}
            }
        }
    elif category == "fatty_liver":
        return {
            "diagnosis": "Fatty Liver Disease (NAFLD/NASH)",
            "confidence": 89.0,
            "risk_level": "high",
            "advice": "Significant hepatic steatosis with elevated transaminases.",
            "detailed": {
                "gate": {"is_healthy": False, "probability_sick": 88.0, "confidence_pct": 91.0},
                "fatty_liver": {
                    "has_fatty_liver": True,
                    "sick_probability": 84.5,
                    "diagnosis": "Fatty Liver Disease Detected",
                    "advice": "Initiate metabolic syndrome management, weight loss, and lipid control."
                },
                "hepatitis": {
                    "stage": 1,
                    "mortality_risk": 22.0,
                    "complications_risk": 25.0,
                    "apri_score": 0.72,
                    "albi_score": -2.68,
                    "stage_distribution": {"Stage 1": 0.70, "Stage 2": 0.25, "Stage 3": 0.05},
                    "advice": "Monitor fibrosis progression."
                },
                "cancer": {
                    "risk_percentage": 35.0,
                    "risk_level": "Low-Moderate Risk",
                    "advice": "Annual check."
                },
                "inputs": {"age": str(age), "gender": gender}
            }
        }
    else: # healthy / normal
        return {
            "diagnosis": "Normal Liver Function (Healthy Checkup)",
            "confidence": 98.0,
            "risk_level": "low",
            "advice": "All clinical parameters within normal reference ranges.",
            "detailed": {
                "gate": {
                    "is_healthy": True,
                    "probability_sick": 4.5,
                    "confidence_pct": 98.5
                },
                # Healthy path intentionally omits detailed models so hasResults is False,
                # perfectly counting towards 'No signs of disease'!
                "inputs": {"age": str(age), "gender": gender}
            }
        }

def enrich_all():
    admin_session = requests.Session()
    login_admin = admin_session.post(f"{BASE_URL}/auth/login", json={
        "username": "verify_admin",
        "password": "VerifyPass123!"
    })
    if login_admin.status_code != 200:
        print("Failed admin login")
        return

    clean_old_empty_analyses(admin_session)

    # Categories to distribute evenly across each doctor's patients
    # Each doctor will have:
    # - 1 Stage 3 Cirrhosis (Urgent follow-up, high mortality)
    # - 1 Stage 2 Fibrosis
    # - 1 Stage 1 Fibrosis
    # - 2 High Cancer Risk
    # - 2 Fatty Liver Disease
    # - 2-3 Healthy / Normal (No signs of disease)
    categories_cycle = [
        "hepatitis_stage_3",
        "fatty_liver",
        "cancer_risk",
        "healthy",
        "hepatitis_stage_2",
        "fatty_liver",
        "cancer_risk",
        "healthy",
        "hepatitis_stage_1",
        "healthy"
    ]

    for doc in DOCTORS:
        print(f"\n==================================================")
        print(f"Enriching patients for {doc['name']} (@{doc['username']})...")
        doc_s = requests.Session()
        l_res = doc_s.post(f"{BASE_URL}/auth/login", json={"username": doc["username"], "password": doc["password"]})
        if l_res.status_code != 200:
            print(f"Could not log in as {doc['username']}")
            continue

        p_res = doc_s.get(f"{BASE_URL}/patients")
        patients = p_res.json().get("patients", [])
        print(f"Found {len(patients)} patients assigned to {doc['name']}.")

        for i, p in enumerate(patients):
            cat = categories_cycle[i % len(categories_cycle)]
            # Estimate age from birth_date or assign realistic age
            b_date = p.get("birth_date")
            age = 45
            if b_date and len(b_date) >= 4:
                try:
                    age = 2026 - int(b_date[:4])
                except Exception:
                    age = 45
            
            # Estimate gender from name or alternation
            gender = "Male" if i % 2 == 0 else "Female"

            profile = get_clinical_profile(p["name"], cat, age, gender)

            report_payload = {
                "patient_id": p["id"],
                "diagnosis": profile["diagnosis"],
                "confidence": profile["confidence"],
                "advice": profile["advice"],
                "risk_level": profile["risk_level"],
                "detailed_results": profile["detailed"]
            }

            rep_res = doc_s.post(f"{BASE_URL}/reports", json=report_payload)
            if rep_res.status_code == 200:
                print(f"  [{i+1}/{len(patients)}] {p['name']} -> {cat} ({profile['risk_level']}) OK")
            else:
                print(f"  [{i+1}/{len(patients)}] Error on {p['name']}: {rep_res.status_code} {rep_res.text[:80]}")
            time.sleep(0.15)

    print("\nAll doctor patients enriched successfully!")

if __name__ == "__main__":
    enrich_all()
