import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import uuid
from dotenv import load_dotenv
from typing import Optional, List, Union, Any, Dict
import json
import requests
from datetime import datetime
import numpy as np

# Load environment variables
load_dotenv()

from diagnosis_engine import DiagnosisEngine
from database import get_db, engine, Base
from models import Patient, LabTest, MedicalReport, User, AuditLog, Medication, Prescription, PrescriptionItem, ClinicalNote, UltrasoundExam, ClinicBilling, ALL_PERMISSIONS, DEFAULT_PERMISSIONS, DOCTOR_PRESET_PERMISSIONS
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    set_auth_cookies, clear_auth_cookies,
    get_current_user, require_permission,
    log_audit, verify_csrf,
)

def make_serializable(obj):
    """Recursively convert NumPy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        # NumPy scalar (works for all NumPy versions)
        return obj.item()
    elif isinstance(obj, (int, float, bool)):
        return obj
    elif hasattr(obj, 'tolist'):
        # NumPy array
        return make_serializable(obj.tolist())
    return obj


def _is_admin(user: User) -> bool:
    """Check if a user has admin-level access (bypasses doctor scoping)."""
    return user.role == "admin" or user.has_permission("can_access_admin")


# Initialize DiagnosisEngine
diagnosis_engine = DiagnosisEngine()

print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
print("Backend server starting with updated analysis saving functionality...")

# Create database tables
Base.metadata.create_all(bind=engine)

# Safe SQLite migrations for columns added to existing tables
try:
    with engine.connect() as conn:
        from sqlalchemy import text
        res = conn.execute(text("PRAGMA table_info(patients)")).fetchall()
        col_names = [r[1] for r in res]
        if "chronic_conditions" not in col_names:
            conn.execute(text("ALTER TABLE patients ADD COLUMN chronic_conditions TEXT"))
            conn.commit()
            print("Migrated: added chronic_conditions column to patients table")

        user_res = conn.execute(text("PRAGMA table_info(users)")).fetchall()
        user_cols = [r[1] for r in user_res]
        if "subscription_plan" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN subscription_plan VARCHAR(50) DEFAULT 'monthly'"))
        if "subscription_months" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN subscription_months INTEGER DEFAULT 1"))
        if "subscription_price" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN subscription_price FLOAT DEFAULT 20.0"))
        if "subscription_expires_at" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN subscription_expires_at DATETIME"))
        conn.commit()
except Exception as e:
    print(f"Migration notice: {e}")

# Initialize FastAPI app
app = FastAPI(title="Medical AI Backend", version="1.0.0")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"VALIDATION ERROR DETAILS: {exc.errors()}") # Prints exact failure to console
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploads directory for imaging / ultrasound scans
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(os.path.join(UPLOAD_DIR, "imaging"), exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Pydantic models
class LabValues(BaseModel):
    ALT: Optional[float] = None
    AST: Optional[float] = None
    Bilirubin: Optional[float] = None
    GGT: Optional[float] = None


class PatientData(BaseModel):
    id: str
    name: str
    age: int
    gender: str

class LabTestResponse(BaseModel):
    testName: str
    value: float
    unit: str
    normalRange: str
    status: str
    date: str

class Visit(BaseModel):
    date: str
    type: str
    doctor: str

class PatientResponse(BaseModel):
    success: bool
    patient: PatientData
    labTests: List[LabTestResponse]
    recentVisits: List[Visit]

class SaveReportRequest(BaseModel):
    patient_id: Any
    diagnosis: Any
    confidence: Any
    advice: Any
    risk_level: Any
    detailed_results: Any

# Initialize Hugging Face API
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

@app.get("/")
async def root():
    return {"message": "Medical AI Backend API", "status": "running"}

@app.post("/analyze")
async def analyze_patient(user_profile: dict, db: Session = Depends(get_db), current_user: User = require_permission("can_run_analysis")):
    try:
        print(f"DEBUG: Received Profile: {user_profile}")

        # Get analysis mode, default to 'full' for backward compatibility
        mode = user_profile.get('mode', 'full')

        # Extract the actual profile data (nested under 'user_profile' if present)
        profile = user_profile.get('user_profile', user_profile)

        # 1. Run Logic based on mode
        if mode == 'gate':
            raw_result = diagnosis_engine.predict_gate_only(profile)
        elif mode == 'cancer':
            raw_result = diagnosis_engine.predict_cancer_only(profile)
        elif mode == 'fatty_liver':
            raw_result = diagnosis_engine.predict_fatty_liver_only(profile)
        elif mode == 'hepatitis':
            raw_result = diagnosis_engine.predict_hepatitis_only(profile)
        else:
            # Default to 'full' for backward compatibility
            raw_result = diagnosis_engine.predict_full_diagnosis(profile)

        # 2. Sanitize for JSON (The Fix)
        clean_result = make_serializable(raw_result)

        # 3. Handle individual model results vs full diagnosis results
        if 'analysis_type' in clean_result:
            # Individual model result (cancer, fatty_liver, hepatitis)
            analysis_type = clean_result['analysis_type']
            model_results = clean_result['results']

            # Extract diagnosis info from the specific model result
            if analysis_type == 'cancer':
                diagnosis = f"Cancer Risk: {model_results['cancer']['risk_level']}"
                confidence = int(model_results['cancer']['risk_percentage'])
                advice = model_results['cancer']['advice']
                risk_level = 'high' if model_results['cancer']['risk_percentage'] > 70 else 'moderate'
            elif analysis_type == 'fatty_liver':
                diagnosis = model_results['fatty_liver']['diagnosis']
                confidence = model_results['fatty_liver']['sick_probability']
                advice = model_results['fatty_liver']['advice']
                risk_level = 'high' if model_results['fatty_liver']['has_fatty_liver'] else 'low'
            elif analysis_type == 'hepatitis':
                diagnosis = f"Hepatitis Stage {model_results['hepatitis']['stage']}"
                confidence = int(model_results['hepatitis']['mortality_risk'])
                advice = model_results['hepatitis']['advice']
                risk_level = model_results['hepatitis']['risk_level']

            detailed_results = model_results
            gate_prediction = 0  # Not applicable for individual models
        else:
            # Full diagnosis or gate result
            diagnosis = clean_result['diagnosis']
            confidence = clean_result['confidence']
            advice = clean_result['advice']
            risk_level = clean_result['risk_level']
            detailed_results = clean_result.get('detailed_results', {})
            gate_prediction = clean_result.get('detailed_results', {}).get('gate', {}).get('prediction', 0)

        # 4. Save to database if patient_id provided (only for full/gate results)
        patient_id_in_data = user_profile.get('patient_id')
        if patient_id_in_data and 'analysis_type' not in clean_result:
            try:
                # Find the patient by database ID (not patient_id string)
                patient = db.query(Patient).filter(Patient.id == patient_id_in_data).first()
                if patient:
                    # Map risk level from DiagnosisEngine to database format
                    risk_level_map = {'low': 'low', 'moderate': 'medium', 'high': 'high'}
                    db_risk_level = risk_level_map.get(risk_level, 'medium')

                    # Save the analysis to database
                    medical_report = MedicalReport(
                        patient_id=patient.id,
                        doctor_id=current_user.id,
                        diagnosis=diagnosis,
                        confidence=confidence,
                        advice=advice,
                        risk_level=db_risk_level,
                        detailed_results=json.dumps(detailed_results)
                    )
                    db.add(medical_report)
                    db.commit()
                    db.refresh(medical_report)
                    print(f"Saved analysis for patient {patient.name} (ID: {patient.id}) - Risk: {db_risk_level}")

                    # ─────────────────────────────────────────────────────
                    # NEW: Save Lab Parameters
                    # ─────────────────────────────────────────────────────
                    try:
                        # Define keys to exclude (demographics, mode, etc.)
                        exclude_keys = {'mode', 'user_profile', 'patient_id', 'age', 'gender', 'name', 'department', 'doctor_id'}
                        
                        # Iterate through input profile data
                        for key, value in profile.items():
                            if key not in exclude_keys and value is not None:
                                # Determine status based on value (simplified logic)
                                # In a real app, you'd compare against specific ranges for each test
                                status = "normal" 
                                normal_range = "N/A"
                                unit = "N/A"

                                # Create LabTest record
                                lab_test = LabTest(
                                    patient_id=patient.id,
                                    test_name=key.replace('_', ' ').title(),
                                    value=float(value) if isinstance(value, (int, float, str)) and str(value).replace('.','',1).isdigit() else 0.0,
                                    unit=unit,
                                    normal_range=normal_range,
                                    status=status,
                                    date=func.now()
                                )
                                db.add(lab_test)
                        
                        db.commit()
                        print(f"Saved lab tests for patient {patient.name}")
                    except Exception as lab_error:
                        print(f"Error saving lab tests: {lab_error}")
                        db.rollback() # Rollback only the lab test part if it fails, though report is already committed
                else:
                    print(f"Patient with ID {patient_id_in_data} not found, analysis not saved")
            except Exception as save_error:
                print(f"Error saving analysis: {save_error}")
                # Don't fail the analysis if saving fails, just log it

        # 5. Return sanitized result
        return {
            "success": True,
            "gate_prediction": gate_prediction,
            "results": detailed_results,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "advice": advice,
            "risk_level": risk_level,
            # Legacy fields for backward compatibility
            "analysis": {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "advice": advice,
                "overallAssessment": f"Risk Level: {risk_level.title()}",
                "recommendations": [advice],
                "detailedAnalyses": detailed_results,
                "scanType": "AI-Powered Multi-Stage Liver Disease Analysis",
                "findings": [
                    {
                        "region": "Liver",
                        "condition": diagnosis,
                        "confidence": confidence / 100.0,
                        "description": advice
                    }
                ],
                "timestamp": datetime.now().isoformat(),
            }
        }

    except Exception as e:
        error_message = str(e)
        # Sanitize error message to remove emojis and problematic characters
        import re
        clean_error = re.sub(r'[^\x00-\x7F]+', '', error_message)  # Remove non-ASCII characters
        print(f"CRITICAL BACKEND CRASH:\n{clean_error}")
        # Return a 400 so the Frontend sees the actual error message
        raise HTTPException(status_code=400, detail=error_message)

@app.post("/reports")
async def create_report(report: SaveReportRequest, db: Session = Depends(get_db), current_user: User = require_permission("can_run_analysis")):
    try:
        print(f"DEBUG: Received Report Payload: {report}")
        # Check if patient exists
        patient_id = report.patient_id
        if not patient_id:
            raise HTTPException(status_code=400, detail="Patient ID required")
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Sanitize the inputs to remove NumPy types
        clean_results = make_serializable(report.detailed_results)

        print("DEBUG: Attempting to add to DB session...")
        # Create new report
        new_report = MedicalReport(
            patient_id=patient_id,
            doctor_id=current_user.id,
            diagnosis=report.diagnosis,
            confidence=float(report.confidence),  # Explicit cast to float
            advice=report.advice,
            risk_level=report.risk_level,
            detailed_results=json.dumps(clean_results)  # Save the SANITIZED version
        )
        db.add(new_report)
        print("DEBUG: Committing to Database...")
        try:
            db.commit()
        except Exception as e:
            print(f"DB ERROR: {str(e)}")
            raise
        db.refresh(new_report)

        return {"success": True, "report_id": new_report.id}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR SAVING REPORT: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/patient-data")
async def get_patient_data(db: Session = Depends(get_db), current_user: User = require_permission("can_view_patients")):
    try:
        # Get the first patient scoped to this doctor
        query = db.query(Patient)
        if not _is_admin(current_user):
            query = query.filter(Patient.doctor_id == current_user.id)
        patient = query.first()

        if not patient:
            # Return mock data if no patients in database
            return {
                "success": True,
                "patient": {
                    "id": "P-2024-001",
                    "name": "John Smith",
                    "age": 45,
                    "gender": "Male",
                },
                "labTests": [
                    {
                        "testName": "Blood Glucose",
                        "value": 95,
                        "unit": "mg/dL",
                        "normalRange": "70-100",
                        "status": "normal",
                        "date": "2024-01-15",
                    },
                    {
                        "testName": "Cholesterol",
                        "value": 185,
                        "unit": "mg/dL",
                        "normalRange": "< 200",
                        "status": "normal",
                        "date": "2024-01-15",
                    },
                ],
                "recentVisits": [
                    {
                        "date": "2024-01-15",
                        "type": "Routine Checkup",
                        "doctor": "Dr. Sarah Ahmed",
                    },
                ],
            }

        # Get lab tests for the patient
        lab_tests = db.query(LabTest).filter(LabTest.patient_id == patient.id).order_by(desc(LabTest.date)).limit(10).all()

        # Convert to response format
        lab_tests_response = [
            {
                "testName": test.test_name,
                "value": test.value,
                "unit": test.unit,
                "normalRange": test.normal_range,
                "status": test.status,
                "date": test.date.isoformat() if test.date else None,
            }
            for test in lab_tests
        ]

        return {
            "success": True,
            "patient": {
                "id": patient.patient_id,
                "name": patient.name,
                "birth_date": patient.birth_date,
            },
            "labTests": lab_tests_response,
            "recentVisits": [
                {
                    "date": "2024-01-15",  # This would come from a visits table in a real implementation
                    "type": "Routine Checkup",
                    "doctor": "Dr. Sarah Ahmed",
                },
            ],
        }
    except Exception as e:
        print(f"Database error in get_patient_data: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/lab-tests")
async def get_lab_tests(patientId: str, db: Session = Depends(get_db), current_user: User = require_permission("can_view_patients")):
    try:
        # Find patient by patient ID
        patient = db.query(Patient).filter(Patient.patient_id == patientId).first()

        if not patient:
            return {"success": False, "message": "Patient not found"}

        # Get lab tests for the patient
        lab_tests = db.query(LabTest).filter(LabTest.patient_id == patient.id).order_by(desc(LabTest.date)).all()

        # Convert to response format
        lab_tests_response = [
            {
                "testName": test.test_name,
                "value": test.value,
                "unit": test.unit,
                "normalRange": test.normal_range,
                "status": test.status,
                "date": test.date.isoformat() if test.date else None,
            }
            for test in lab_tests
        ]

        return {
            "success": True,
            "labTests": lab_tests_response,
        }
    except Exception as e:
        print(f"Database error in get_lab_tests: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/patients")
async def get_patients(request: Request, db: Session = Depends(get_db), current_user: User = require_permission("can_view_patients")):
    try:
        # Parse query parameters manually
        query_params = dict(request.query_params)
        status = query_params.get('status', 'active')

        # Filter patients by status (default to active)
        query = db.query(Patient)
        if status != "all":
            query = query.filter(Patient.status == status)

        # Doctor scoping: non-admin users only see their own patients
        if not _is_admin(current_user):
            query = query.filter(Patient.doctor_id == current_user.id)

        patients = query.order_by(desc(Patient.created_at)).all()

        patients_response = [
            {
                "id": patient.id,
                "name": patient.name,
                "patient_id": patient.patient_id,
                "birth_date": patient.birth_date,
                "email": patient.email,
                "phone": patient.phone,
                "profile_picture": patient.profile_picture,
                "department": patient.department,
                "doctor_name": patient.doctor_name,
                "status": patient.status,
                "chronic_conditions": patient.chronic_conditions,
                "created_at": patient.created_at.isoformat() if patient.created_at else None,
                "updated_at": patient.updated_at.isoformat() if patient.updated_at else None,
            }
            for patient in patients
        ]

        return {
            "success": True,
            "patients": patients_response,
        }
    except Exception as e:
        print(f"Database error in get_patients: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.put("/patients/{patient_id}/archive")
async def archive_patient(patient_id: str, db: Session = Depends(get_db), current_user: User = require_permission("can_edit_patients")):
    try:
        # Find the patient by database ID (integer)
        patient_id_int = int(patient_id)
        patient = db.query(Patient).filter(Patient.id == patient_id_int).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Ownership check
        if not _is_admin(current_user) and patient.doctor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not your patient")

        if patient.status == "archived":
            raise HTTPException(status_code=400, detail="Patient is already archived")

        # Archive the patient and analyses
        patient.status = "archived"
        analyses_to_archive = db.query(MedicalReport).filter(MedicalReport.patient_id == patient.id).all()
        for analysis in analyses_to_archive:
            analysis.status = "archived"

        db.commit()

        return {"success": True, "message": f"Patient and {len(analyses_to_archive)} analyses archived successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error in archive_patient: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.put("/patients/{patient_id}/restore")
async def restore_patient(patient_id: str, db: Session = Depends(get_db), current_user: User = require_permission("can_edit_patients")):
    try:
        # Find the patient by database ID (integer)
        patient_id_int = int(patient_id)
        patient = db.query(Patient).filter(Patient.id == patient_id_int).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Ownership check
        if not _is_admin(current_user) and patient.doctor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not your patient")

        if patient.status == "active":
            raise HTTPException(status_code=400, detail="Patient is already active")

        # Restore the patient and analyses
        patient.status = "active"
        analyses_to_restore = db.query(MedicalReport).filter(
            MedicalReport.patient_id == patient.id,
            MedicalReport.status == "archived"
        ).all()
        for analysis in analyses_to_restore:
            analysis.status = "active"

        db.commit()

        return {"success": True, "message": f"Patient and {len(analyses_to_restore)} analyses restored successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error in restore_patient: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.put("/patients/{patient_id}")
async def update_patient(patient_id: str, patient_data: dict = None, db: Session = Depends(get_db), current_user: User = require_permission("can_edit_patients")):
    try:
        # Find patient by database ID (integer)
        patient_id_int = int(patient_id)
        patient = db.query(Patient).filter(Patient.id == patient_id_int).first()

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Ownership check
        if not _is_admin(current_user) and patient.doctor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not your patient")

        # Update patient fields
        if "name" in patient_data:
            patient.name = patient_data["name"]
        if "patient_id" in patient_data:
            patient.patient_id = patient_data["patient_id"]
        if "birth_date" in patient_data:
            patient.birth_date = patient_data["birth_date"]
        if "email" in patient_data:
            patient.email = patient_data["email"]
        if "phone" in patient_data:
            patient.phone = patient_data["phone"]
        if "profile_picture" in patient_data:
            patient.profile_picture = patient_data["profile_picture"]
        if "department" in patient_data:
            patient.department = patient_data["department"]
        if "doctor_name" in patient_data:
            patient.doctor_name = patient_data["doctor_name"]
        if "chronic_conditions" in patient_data:
            c_val = patient_data["chronic_conditions"]
            patient.chronic_conditions = json.dumps(c_val) if isinstance(c_val, (list, dict)) else c_val

        db.commit()
        db.refresh(patient)

        return {"success": True, "patient": {
            "id": patient.id,
            "name": patient.name,
            "patient_id": patient.patient_id,
            "birth_date": patient.birth_date,
            "email": patient.email,
            "phone": patient.phone,
            "profile_picture": patient.profile_picture,
            "department": patient.department,
            "doctor_name": patient.doctor_name,
            "status": patient.status
        }, "message": "Patient updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error in update_patient: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.delete("/patients/{patient_id}")
async def delete_patient(patient_id: str, db: Session = Depends(get_db), current_user: User = require_permission("can_delete_patients")):
    try:
        # Find the patient by database ID (integer)
        patient_id_int = int(patient_id)
        patient = db.query(Patient).filter(Patient.id == patient_id_int).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Ownership check
        if not _is_admin(current_user) and patient.doctor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not your patient")

        if patient.status == "active":
            raise HTTPException(status_code=400, detail="Cannot delete active patient. Archive first.")

        # Permanently delete the patient and analyses
        # CASCADE DELETE: Remove reports first
        db.query(MedicalReport).filter(MedicalReport.patient_id == patient.id).delete()
        # NOW delete the patient
        db.delete(patient)

        db.commit()

        return {"success": True, "message": "Patient and all associated analyses permanently deleted"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error in delete_patient: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/patients")
async def create_or_update_patient(patient_data: dict, db: Session = Depends(get_db), current_user: User = require_permission("can_create_patients")):
    try:
        patient_id = patient_data.get("patient_id")
        name = patient_data.get("name")
        birth_date_str = patient_data.get("birth_date")

        if not patient_id or not name:
            raise HTTPException(status_code=400, detail="Patient ID and name are required")

        # Check if patient already exists
        existing_patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        if existing_patient:
            # Return error for duplicate patient ID
            raise HTTPException(status_code=400, detail="ID is Currently used")

        # birth_date is now stored as string
        birth_date = birth_date_str if birth_date_str else None

        # Create new patient (auto-assign to creating doctor)
        new_patient = Patient(
            name=name,
            birth_date=birth_date,
            patient_id=patient_id,
            doctor_id=current_user.id,
            email=patient_data.get("email"),
            phone=patient_data.get("phone"),
            profile_picture=patient_data.get("profile_picture"),
            department=patient_data.get("department"),
            doctor_name=patient_data.get("doctor_name")
        )

        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)

        return {"success": True, "patient": {
            "id": new_patient.id,
            "name": new_patient.name,
            "patient_id": new_patient.patient_id,
            "birth_date": new_patient.birth_date,
            "email": new_patient.email,
            "phone": new_patient.phone,
            "profile_picture": new_patient.profile_picture,
            "department": new_patient.department,
            "doctor_name": new_patient.doctor_name
        }}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error in create_or_update_patient: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/patient-analyses")
async def get_patient_analyses(request: Request, db: Session = Depends(get_db), current_user: User = require_permission("can_view_reports")):
    try:
        # Parse query parameters manually
        query_params = dict(request.query_params)
        include_archived = query_params.get('include_archived', 'false').lower() == 'true'
        patient_id_filter = query_params.get('patient_id')

        # Get medical reports - filter out archived by default unless specifically requested
        query = db.query(MedicalReport)
        if not include_archived:
            query = query.filter(MedicalReport.status != "archived")

        # Doctor scoping: non-admin users only see analyses for their own patients
        if not _is_admin(current_user):
            doctor_patient_ids = [
                p.id for p in db.query(Patient.id).filter(Patient.doctor_id == current_user.id).all()
            ]
            query = query.filter(MedicalReport.patient_id.in_(doctor_patient_ids))

        # Filter by patient_id if provided (matches patient.patient_id string field)
        if patient_id_filter:
            # Find patient by patient_id string
            patient = db.query(Patient).filter(Patient.patient_id == patient_id_filter).first()
            if patient:
                query = query.filter(MedicalReport.patient_id == patient.id)

        analyses = query.order_by(desc(MedicalReport.created_at)).all()

        # Convert to response format
        analyses_response = []
        for analysis in analyses:
            # Explicitly fetch the patient
            patient = db.query(Patient).filter(Patient.id == analysis.patient_id).first()

            # Calculate risk level if not set
            risk_level = analysis.risk_level
            if not risk_level:
                if analysis.confidence >= 80:
                    risk_level = "low"
                elif analysis.confidence >= 60:
                    risk_level = "medium"
                else:
                    risk_level = "high"

            # Look up doctor full name from User table
            doctor_display_name = patient.doctor_name if patient else None
            if patient and patient.doctor_id:
                doctor_user = db.query(User).filter(User.id == patient.doctor_id).first()
                if doctor_user and doctor_user.full_name:
                    doctor_display_name = doctor_user.full_name

            patient_data = {
                "id": analysis.id,
                "patient_id": analysis.patient_id,
                "diagnosis": analysis.diagnosis,
                "confidence": analysis.confidence,
                "advice": analysis.advice,
                "status": analysis.status,
                "is_finalized": bool(analysis.is_finalized),
                "risk_level": risk_level,
                "detailed_results": analysis.detailed_results,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                "updated_at": analysis.updated_at.isoformat() if analysis.updated_at else analysis.created_at.isoformat() if analysis.created_at else None,
                "patient_name": patient.name if patient else "Unknown",
                "patient_id_display": patient.patient_id if patient else "Unknown",
                "birth_date": patient.birth_date if patient else None,
                "email": patient.email if patient else None,
                "phone": patient.phone if patient else None,
                "profile_picture": patient.profile_picture if patient else None,
                "department": patient.department if patient else None,
                "doctor_name": doctor_display_name,
            }
            analyses_response.append(patient_data)

        return {
            "success": True,
            "analyses": analyses_response,
        }
    except Exception as e:
        print(f"Database error in get_patient_analyses: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.put("/patient-analyses/{analysis_id}")
async def update_patient_analysis(analysis_id: int, analysis_data: dict, db: Session = Depends(get_db), current_user: User = require_permission("can_edit_patients")):
    try:
        analysis = db.query(MedicalReport).filter(MedicalReport.id == analysis_id).first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Update fields
        if "diagnosis" in analysis_data:
            analysis.diagnosis = analysis_data["diagnosis"]
        if "confidence" in analysis_data:
            analysis.confidence = analysis_data["confidence"]
        if "advice" in analysis_data:
            analysis.advice = analysis_data["advice"]
        if "risk_level" in analysis_data:
            analysis.risk_level = analysis_data["risk_level"]
        if "detailed_results" in analysis_data:
            val = analysis_data["detailed_results"]
            analysis.detailed_results = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
        if "created_at" in analysis_data and analysis_data["created_at"]:
            try:
                analysis.created_at = datetime.fromisoformat(analysis_data["created_at"])
            except Exception:
                pass
        if "patient_id" in analysis_data:
            # Verify that the patient exists
            patient = db.query(Patient).filter(Patient.id == analysis_data["patient_id"]).first()
            if not patient:
                raise HTTPException(status_code=400, detail="Patient not found")
            analysis.patient_id = analysis_data["patient_id"]

        db.commit()
        db.refresh(analysis)

        return {"success": True, "analysis": {
            "id": analysis.id,
            "patient_id": analysis.patient_id,
            "diagnosis": analysis.diagnosis,
            "confidence": analysis.confidence,
            "advice": analysis.advice
        }}

    except Exception as e:
        print(f"Database error in update_patient_analysis: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.delete("/patient-analyses/{analysis_id}")
async def delete_patient_analysis(analysis_id: int, db: Session = Depends(get_db), current_user: User = require_permission("can_delete_patients")):
    try:
        analysis = db.query(MedicalReport).filter(MedicalReport.id == analysis_id).first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        db.delete(analysis)
        db.commit()

        return {"success": True, "message": "Analysis deleted successfully"}

    except Exception as e:
        print(f"Database error in delete_patient_analysis: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# ═══════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    role: Optional[str] = "user"
    permissions: Optional[Dict[str, bool]] = None
    subscription_plan: Optional[str] = "monthly"
    subscription_months: Optional[int] = 1
    subscription_price: Optional[float] = 20.0


@app.post("/auth/login")
async def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    """Authenticate user, set JWT cookies, return user info + permissions."""
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # Create tokens
    access_token = create_access_token({"sub": str(user.id)})
    refresh_token = create_refresh_token({"sub": str(user.id)})

    # Update last_login
    user.last_login = datetime.utcnow()
    db.commit()

    # Audit log
    client_ip = request.client.host if request.client else None
    log_audit(db, user.id, "LOGIN", ip_address=client_ip)

    # Build response with cookies
    response = JSONResponse(content={
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "fullName": user.full_name or user.username,
            "role": user.role,
            "permissions": user.get_permissions(),
        },
    })
    set_auth_cookies(response, access_token, refresh_token)
    return response


@app.post("/auth/logout")
async def logout(request: Request, db: Session = Depends(get_db)):
    """Clear auth cookies."""
    response = JSONResponse(content={"success": True, "message": "Logged out"})
    clear_auth_cookies(response)

    # Try to log audit if user was authenticated
    try:
        token = request.cookies.get("access_token")
        if token:
            payload = decode_token(token)
            user_id = payload.get("sub")
            if user_id:
                client_ip = request.client.host if request.client else None
                log_audit(db, int(user_id), "LOGOUT", ip_address=client_ip)
    except Exception:
        pass  # Don't fail logout if audit logging fails

    return response


@app.post("/auth/refresh")
async def refresh_token(request: Request, db: Session = Depends(get_db)):
    """Refresh access token using refresh token cookie."""
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")

    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or deactivated")

    new_access = create_access_token({"sub": str(user.id)})
    new_refresh = create_refresh_token({"sub": str(user.id)})

    response = JSONResponse(content={"success": True})
    set_auth_cookies(response, new_access, new_refresh)
    return response


@app.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the current authenticated user's info + permissions."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "fullName": current_user.full_name or current_user.username,
        "role": current_user.role,
        "permissions": current_user.get_permissions(),
    }


@app.post("/auth/register")
async def register_user(
    body: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_users"),
):
    """Admin-only: create a new user account."""
    # Check duplicates
    existing = db.query(User).filter(
        (User.username == body.username) | (User.email == body.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    # Pick the right default permissions based on role
    if body.permissions:
        perms = body.permissions
    elif body.role == "doctor":
        perms = DOCTOR_PRESET_PERMISSIONS.copy()
    else:
        perms = DEFAULT_PERMISSIONS.copy()

    # Safety: only admin role may have can_access_admin
    if (body.role or "user") != "admin":
        perms["can_access_admin"] = False

    from datetime import timedelta
    sub_months = body.subscription_months or 1
    sub_expires = datetime.utcnow() + timedelta(days=sub_months * 30)

    new_user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role or "user",
        is_active=1,
        permissions=json.dumps(perms),
        subscription_plan=body.subscription_plan or "monthly",
        subscription_months=sub_months,
        subscription_price=body.subscription_price if body.subscription_price is not None else 20.0,
        subscription_expires_at=sub_expires,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Audit
    client_ip = request.client.host if request.client else None
    log_audit(db, current_user.id, "CREATE_USER", resource="users",
              resource_id=new_user.id, ip_address=client_ip)

    return {
        "success": True,
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "fullName": new_user.full_name,
            "role": new_user.role,
            "permissions": new_user.get_permissions(),
            "subscriptionPlan": new_user.subscription_plan,
            "subscriptionMonths": new_user.subscription_months,
            "subscriptionPrice": new_user.subscription_price,
            "subscriptionExpiresAt": new_user.subscription_expires_at.isoformat() if new_user.subscription_expires_at else None,
        },
    }


# ═══════════════════════════════════════════════════════════
# ADMIN ROUTES
# ═══════════════════════════════════════════════════════════

@app.get("/admin/users")
async def admin_list_users(
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_users"),
):
    """List all users with their permissions."""
    users = db.query(User).order_by(desc(User.created_at)).all()
    return {
        "success": True,
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "fullName": u.full_name,
                "role": u.role,
                "isActive": bool(u.is_active),
                "permissions": u.get_permissions(),
                "subscriptionPlan": getattr(u, "subscription_plan", "monthly") or "monthly",
                "subscriptionMonths": getattr(u, "subscription_months", 1) or 1,
                "subscriptionPrice": getattr(u, "subscription_price", 20.0) or 20.0,
                "subscriptionExpiresAt": u.subscription_expires_at.isoformat() if getattr(u, "subscription_expires_at", None) else None,
                "lastLogin": u.last_login.isoformat() if u.last_login else None,
                "createdAt": u.created_at.isoformat() if u.created_at else None,
            }
            for u in users
        ],
    }


@app.put("/admin/users/{user_id}")
async def admin_update_user(
    user_id: int,
    user_data: dict,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_users"),
):
    """Update a user's profile, role, active status, or permissions."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent admin from deactivating themselves
    if user.id == current_user.id and "is_active" in user_data and not user_data["is_active"]:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    if "full_name" in user_data:
        user.full_name = user_data["full_name"]
    if "email" in user_data:
        user.email = user_data["email"]
    if "role" in user_data:
        user.role = user_data["role"]
    if "is_active" in user_data:
        user.is_active = 1 if user_data["is_active"] else 0
    if "permissions" in user_data:
        user.permissions = json.dumps(user_data["permissions"])
    if "password" in user_data and user_data["password"]:
        user.hashed_password = hash_password(user_data["password"])
    if "subscription_plan" in user_data:
        user.subscription_plan = user_data["subscription_plan"]
    if "subscription_months" in user_data:
        user.subscription_months = int(user_data["subscription_months"])
        from datetime import timedelta
        user.subscription_expires_at = datetime.utcnow() + timedelta(days=user.subscription_months * 30)
    if "subscription_price" in user_data:
        user.subscription_price = float(user_data["subscription_price"])

    db.commit()
    db.refresh(user)

    # Audit
    client_ip = request.client.host if request.client else None
    log_audit(db, current_user.id, "UPDATE_USER", resource="users",
              resource_id=user.id, ip_address=client_ip)

    return {
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "fullName": user.full_name,
            "role": user.role,
            "isActive": bool(user.is_active),
            "permissions": user.get_permissions(),
            "subscriptionPlan": getattr(user, "subscription_plan", "monthly"),
            "subscriptionMonths": getattr(user, "subscription_months", 1),
            "subscriptionPrice": getattr(user, "subscription_price", 20.0),
            "subscriptionExpiresAt": user.subscription_expires_at.isoformat() if getattr(user, "subscription_expires_at", None) else None,
        },
    }



@app.delete("/admin/users/{user_id}")
async def admin_deactivate_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_users"),
):
    """Deactivate a user (soft delete)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    user.is_active = 0
    db.commit()

    client_ip = request.client.host if request.client else None
    log_audit(db, current_user.id, "DEACTIVATE_USER", resource="users",
              resource_id=user.id, ip_address=client_ip)

    return {"success": True, "message": f"User {user.username} deactivated"}


@app.delete("/admin/users/{user_id}/permanent")
async def admin_delete_user_permanent(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_users"),
):
    """Permanently delete a user (and unassign their patients/reports)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    # 1. Unassign Patients
    db.query(Patient).filter(Patient.doctor_id == user.id).update({"doctor_id": None})
    
    # 2. Unassign Medical Reports
    db.query(MedicalReport).filter(MedicalReport.doctor_id == user.id).update({"doctor_id": None})

    # 3. Create Audit Log (BEFORE deleting user, so we log WHO did it)
    client_ip = request.client.host if request.client else None
    log_audit(db, current_user.id, "DELETE_USER_PERMANENT", resource="users",
              resource_id=user.id, details=f"Deleted user {user.username}", ip_address=client_ip)
    
    # 4. Delete the User
    db.delete(user)
    db.commit()

    return {"success": True, "message": f"User {user.username} permanently deleted. Patients unassigned."}


@app.get("/admin/stats")
async def admin_stats(
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_access_admin"),
):
    """Dashboard overview statistics."""
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == 1).count()
    total_patients = db.query(Patient).filter(Patient.status == "active").count()
    total_analyses = db.query(MedicalReport).filter(MedicalReport.status != "archived").count()

    # Recent logins (last 10)
    recent_logs = (
        db.query(AuditLog)
        .filter(AuditLog.action == "LOGIN")
        .order_by(desc(AuditLog.created_at))
        .limit(10)
        .all()
    )
    recent_logins = []
    for log in recent_logs:
        u = db.query(User).filter(User.id == log.user_id).first()
        recent_logins.append({
            "username": u.username if u else "unknown",
            "timestamp": log.created_at.isoformat() if log.created_at else None,
            "ip": log.ip_address,
        })

    return {
        "success": True,
        "stats": {
            "totalUsers": total_users,
            "activeUsers": active_users,
            "totalPatients": total_patients,
            "totalAnalyses": total_analyses,
            "recentLogins": recent_logins,
        },
    }


@app.get("/admin/audit-logs")
async def admin_audit_logs(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_view_audit_logs"),
):
    """Paginated audit log listing."""
    query_params = dict(request.query_params)
    page = int(query_params.get("page", "1"))
    per_page = int(query_params.get("per_page", "20"))
    action_filter = query_params.get("action")
    user_filter = query_params.get("user_id")

    query = db.query(AuditLog)
    if action_filter:
        query = query.filter(AuditLog.action == action_filter)
    if user_filter:
        query = query.filter(AuditLog.user_id == int(user_filter))

    total = query.count()
    logs = (
        query.order_by(desc(AuditLog.created_at))
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    logs_response = []
    for log in logs:
        u = db.query(User).filter(User.id == log.user_id).first()
        logs_response.append({
            "id": log.id,
            "userId": log.user_id,
            "username": u.username if u else "unknown",
            "action": log.action,
            "resource": log.resource,
            "resourceId": log.resource_id,
            "details": log.details,
            "ipAddress": log.ip_address,
            "createdAt": log.created_at.isoformat() if log.created_at else None,
        })

    return {
        "success": True,
        "total": total,
        "page": page,
        "perPage": per_page,
        "logs": logs_response,
    }


@app.get("/admin/permission-presets")
async def get_permission_presets(
    current_user: User = require_permission("can_manage_users"),
):
    """Return available permission presets for the Admin UI."""
    return {
        "success": True,
        "presets": {
            "default": DEFAULT_PERMISSIONS,
            "doctor": DOCTOR_PRESET_PERMISSIONS,
            "admin": ALL_PERMISSIONS,
        },
    }


# ═══════════════════════════════════════════════════════════
# ADMIN – PATIENT ASSIGNMENT
# ═══════════════════════════════════════════════════════════

@app.get("/admin/patients")
async def admin_list_patients(
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_access_admin"),
):
    """List all patients with their assigned doctor info (admin only)."""
    patients = db.query(Patient).order_by(desc(Patient.created_at)).all()
    patients_response = []
    for p in patients:
        doctor = None
        if p.doctor_id:
            doctor = db.query(User).filter(User.id == p.doctor_id).first()
        patients_response.append({
            "id": p.id,
            "name": p.name,
            "patient_id": p.patient_id,
            "status": p.status,
            "email": p.email,
            "phone": p.phone,
            "doctor_id": p.doctor_id,
            "doctor_username": doctor.username if doctor else None,
            "doctor_full_name": doctor.full_name if doctor else None,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        })
    return {"success": True, "patients": patients_response}


@app.put("/admin/patients/{patient_id}/assign")
async def admin_assign_patient(
    patient_id: int,
    body: dict,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_users"),
):
    """Assign or reassign a patient (and their reports) to a doctor."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    new_doctor_id = body.get("doctor_id")  # None means unassign
    if new_doctor_id is not None:
        doctor = db.query(User).filter(User.id == new_doctor_id).first()
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found")

    old_doctor_id = patient.doctor_id
    patient.doctor_id = new_doctor_id

    # Also reassign all medical reports for this patient
    db.query(MedicalReport).filter(
        MedicalReport.patient_id == patient.id
    ).update({"doctor_id": new_doctor_id})

    db.commit()

    # Audit
    client_ip = request.client.host if request.client else None
    # Audit
    client_ip = request.client.host if request.client else None
    log_audit(
        db, current_user.id, "ASSIGN_PATIENT",
        resource="patients", resource_id=patient.id,
        details={"old_doctor_id": old_doctor_id, "new_doctor_id": new_doctor_id},
        ip_address=client_ip,
    )

    return {"success": True, "message": f"Patient {patient.name} assigned to doctor {new_doctor_id}"}


# ─────────────────────────────────────────────────────────
# Phase 1: E-Prescriptions & Medications Endpoints
# ─────────────────────────────────────────────────────────

class PrescriptionItemCreate(BaseModel):
    medication_id: Optional[int] = None
    medication_name: str
    generic_name: Optional[str] = None
    dose: str
    frequency: str
    timing: Optional[str] = None
    duration: Optional[str] = None
    instructions_ar: Optional[str] = None

class PrescriptionCreate(BaseModel):
    patient_id: int
    diagnosis: Optional[str] = None
    notes: Optional[str] = None
    follow_up_date: Optional[str] = None
    items: List[PrescriptionItemCreate]

class MedicationCreate(BaseModel):
    trade_name: str
    generic_name: str
    category: Optional[str] = None
    dosage_forms: Optional[List[str]] = None
    default_dose: Optional[str] = None
    default_freq: Optional[str] = None
    timing: Optional[str] = None
    notes_ar: Optional[str] = None
    liver_warning: Optional[str] = None
    is_liver_safe: Optional[int] = 1
    common_in_iraq: Optional[int] = 1


@app.get("/medications")
async def get_medications(
    q: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Medication)
    if category:
        query = query.filter(Medication.category == category)
    if q:
        search = f"%{q}%"
        query = query.filter(
            (Medication.trade_name.ilike(search)) | (Medication.generic_name.ilike(search))
        )
    meds = query.order_by(Medication.trade_name.asc()).all()
    
    result = []
    for m in meds:
        dosage_forms_list = []
        if m.dosage_forms:
            try:
                dosage_forms_list = json.loads(m.dosage_forms)
            except Exception:
                dosage_forms_list = [m.dosage_forms]
        result.append({
            "id": m.id,
            "trade_name": m.trade_name,
            "generic_name": m.generic_name,
            "category": m.category,
            "dosage_forms": dosage_forms_list,
            "default_dose": m.default_dose,
            "default_freq": m.default_freq,
            "timing": m.timing,
            "notes_ar": m.notes_ar,
            "liver_warning": m.liver_warning,
            "is_liver_safe": bool(m.is_liver_safe),
            "common_in_iraq": bool(m.common_in_iraq),
        })
    return {"success": True, "medications": result}


@app.post("/medications")
async def create_medication(
    med_in: MedicationCreate,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_prescriptions")
):
    new_med = Medication(
        trade_name=med_in.trade_name,
        generic_name=med_in.generic_name,
        category=med_in.category,
        dosage_forms=json.dumps(med_in.dosage_forms) if med_in.dosage_forms else None,
        default_dose=med_in.default_dose,
        default_freq=med_in.default_freq,
        timing=med_in.timing,
        notes_ar=med_in.notes_ar,
        liver_warning=med_in.liver_warning,
        is_liver_safe=med_in.is_liver_safe if med_in.is_liver_safe is not None else 1,
        common_in_iraq=med_in.common_in_iraq if med_in.common_in_iraq is not None else 1
    )
    db.add(new_med)
    db.commit()
    db.refresh(new_med)
    return {"success": True, "medication_id": new_med.id}


@app.get("/prescriptions")
async def get_prescriptions(
    patient_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Prescription)
    if not _is_admin(current_user):
        query = query.filter(Prescription.doctor_id == current_user.id)
    if patient_id:
        query = query.filter(Prescription.patient_id == patient_id)
        
    prescriptions = query.order_by(desc(Prescription.created_at)).all()
    
    result = []
    for rx in prescriptions:
        patient = rx.patient
        doctor = rx.doctor
        result.append({
            "id": rx.id,
            "prescription_number": rx.prescription_number,
            "patient_id": rx.patient_id,
            "patient_name": patient.name if patient else "Unknown",
            "patient_code": patient.patient_id if patient else "",
            "patient_birth_date": patient.birth_date if patient else None,
            "patient_phone": patient.phone if patient else None,
            "doctor_id": rx.doctor_id,
            "doctor_name": doctor.full_name if doctor and doctor.full_name else (doctor.username if doctor else "Unknown"),
            "diagnosis": rx.diagnosis,
            "notes": rx.notes,
            "follow_up_date": rx.follow_up_date,
            "status": rx.status,
            "item_count": len(rx.items),
            "created_at": rx.created_at.isoformat() if rx.created_at else None,
        })
    return {"success": True, "prescriptions": result}


@app.post("/prescriptions")
async def create_prescription(
    rx_in: PrescriptionCreate,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_prescriptions")
):
    patient = db.query(Patient).filter(Patient.id == rx_in.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    date_str = datetime.now().strftime("%Y%m%d")
    count_today = db.query(Prescription).filter(
        Prescription.prescription_number.like(f"RX-{date_str}-%")
    ).count()
    rx_number = f"RX-{date_str}-{count_today + 1:03d}"

    new_rx = Prescription(
        prescription_number=rx_number,
        patient_id=patient.id,
        doctor_id=current_user.id,
        diagnosis=rx_in.diagnosis,
        notes=rx_in.notes,
        follow_up_date=rx_in.follow_up_date,
        status="active"
    )
    db.add(new_rx)
    db.flush()

    for item in rx_in.items:
        p_item = PrescriptionItem(
            prescription_id=new_rx.id,
            medication_id=item.medication_id,
            medication_name=item.medication_name,
            generic_name=item.generic_name,
            dose=item.dose,
            frequency=item.frequency,
            timing=item.timing,
            duration=item.duration,
            instructions_ar=item.instructions_ar
        )
        db.add(p_item)

    db.commit()
    db.refresh(new_rx)

    return {
        "success": True,
        "prescription_id": new_rx.id,
        "prescription_number": new_rx.prescription_number,
        "message": "Prescription created successfully"
    }


@app.get("/prescriptions/{prescription_id}")
async def get_prescription_by_id(
    prescription_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    rx = db.query(Prescription).filter(Prescription.id == prescription_id).first()
    if not rx:
        raise HTTPException(status_code=404, detail="Prescription not found")
        
    if not _is_admin(current_user) and rx.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    patient = rx.patient
    doctor = rx.doctor
    
    items = []
    for it in rx.items:
        items.append({
            "id": it.id,
            "medication_id": it.medication_id,
            "medication_name": it.medication_name,
            "generic_name": it.generic_name,
            "dose": it.dose,
            "frequency": it.frequency,
            "timing": it.timing,
            "duration": it.duration,
            "instructions_ar": it.instructions_ar,
        })

    return {
        "success": True,
        "prescription": {
            "id": rx.id,
            "prescription_number": rx.prescription_number,
            "patient_id": rx.patient_id,
            "patient_name": patient.name if patient else "Unknown",
            "patient_code": patient.patient_id if patient else "",
            "patient_birth_date": patient.birth_date if patient else None,
            "patient_phone": patient.phone if patient else None,
            "doctor_id": rx.doctor_id,
            "doctor_name": doctor.full_name if doctor and doctor.full_name else (doctor.username if doctor else "Unknown"),
            "diagnosis": rx.diagnosis,
            "notes": rx.notes,
            "follow_up_date": rx.follow_up_date,
            "status": rx.status,
            "created_at": rx.created_at.isoformat() if rx.created_at else None,
            "items": items
        }
    }


@app.delete("/prescriptions/{prescription_id}")
async def delete_prescription(
    prescription_id: int,
    db: Session = Depends(get_db),
    current_user: User = require_permission("can_manage_prescriptions")
):
    rx = db.query(Prescription).filter(Prescription.id == prescription_id).first()
    if not rx:
        raise HTTPException(status_code=404, detail="Prescription not found")
        
    if not _is_admin(current_user) and rx.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    db.delete(rx)
    db.commit()
    return {"success": True, "message": "Prescription deleted"}


# ─────────────────────────────────────────────────────────
# Phase 3: Clinical Notes (SOAP) & Chronic Conditions Endpoints
# ─────────────────────────────────────────────────────────

class ClinicalNoteCreate(BaseModel):
    patient_id: int
    visit_type: Optional[str] = "follow_up" # new, follow_up, routine
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    blood_pressure: Optional[str] = None
    heart_rate: Optional[int] = None
    weight: Optional[float] = None
    temperature: Optional[float] = None
    jaundice: Optional[str] = "None"
    ascites: Optional[str] = "None"
    edema: Optional[str] = "None"
    hepatomegaly: Optional[int] = 0
    splenomegaly: Optional[int] = 0
    spider_angioma: Optional[int] = 0
    asterixis: Optional[int] = 0
    follow_up_date: Optional[str] = None


class ClinicalNoteUpdate(BaseModel):
    visit_type: Optional[str] = None
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    blood_pressure: Optional[str] = None
    heart_rate: Optional[int] = None
    weight: Optional[float] = None
    temperature: Optional[float] = None
    jaundice: Optional[str] = None
    ascites: Optional[str] = None
    edema: Optional[str] = None
    hepatomegaly: Optional[int] = None
    splenomegaly: Optional[int] = None
    spider_angioma: Optional[int] = None
    asterixis: Optional[int] = None
    follow_up_date: Optional[str] = None


def _format_clinical_note(note: ClinicalNote) -> dict:
    patient = note.patient
    doctor = note.doctor
    return {
        "id": note.id,
        "patient_id": note.patient_id,
        "patient_name": patient.name if patient else "Unknown",
        "patient_code": patient.patient_id if patient else "",
        "patient_birth_date": patient.birth_date if patient else None,
        "patient_phone": patient.phone if patient else None,
        "doctor_id": note.doctor_id,
        "doctor_name": doctor.full_name if doctor and doctor.full_name else (doctor.username if doctor else "Attending Physician"),
        "visit_date": note.visit_date.isoformat() if note.visit_date else None,
        "visit_type": note.visit_type or "follow_up",
        "subjective": note.subjective or "",
        "objective": note.objective or "",
        "assessment": note.assessment or "",
        "plan": note.plan or "",
        "blood_pressure": note.blood_pressure or "",
        "heart_rate": note.heart_rate,
        "weight": note.weight,
        "temperature": note.temperature,
        "jaundice": note.jaundice or "None",
        "ascites": note.ascites or "None",
        "edema": note.edema or "None",
        "hepatomegaly": bool(note.hepatomegaly),
        "splenomegaly": bool(note.splenomegaly),
        "spider_angioma": bool(note.spider_angioma),
        "asterixis": bool(note.asterixis),
        "follow_up_date": note.follow_up_date,
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }


@app.get("/clinical-notes")
async def get_clinical_notes(
    patient_id: Optional[Union[int, str]] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(ClinicalNote)
    if not _is_admin(current_user):
        query = query.filter(ClinicalNote.doctor_id == current_user.id)

    if patient_id is not None:
        try:
            pid_int = int(patient_id)
            query = query.filter(ClinicalNote.patient_id == pid_int)
        except ValueError:
            # Match by string patient_id e.g. "P-001"
            p = db.query(Patient).filter(Patient.patient_id == str(patient_id)).first()
            if p:
                query = query.filter(ClinicalNote.patient_id == p.id)
            else:
                return {"success": True, "notes": []}

    notes = query.order_by(desc(ClinicalNote.created_at)).limit(limit).all()
    results = [_format_clinical_note(n) for n in notes]
    return {"success": True, "notes": results}


@app.post("/clinical-notes")
async def create_clinical_note(
    note_in: ClinicalNoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check patient exists
    patient = db.query(Patient).filter(Patient.id == note_in.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    new_note = ClinicalNote(
        patient_id=note_in.patient_id,
        doctor_id=current_user.id,
        visit_type=note_in.visit_type or "follow_up",
        subjective=note_in.subjective,
        objective=note_in.objective,
        assessment=note_in.assessment,
        plan=note_in.plan,
        blood_pressure=note_in.blood_pressure,
        heart_rate=note_in.heart_rate,
        weight=note_in.weight,
        temperature=note_in.temperature,
        jaundice=note_in.jaundice,
        ascites=note_in.ascites,
        edema=note_in.edema,
        hepatomegaly=note_in.hepatomegaly or 0,
        splenomegaly=note_in.splenomegaly or 0,
        spider_angioma=note_in.spider_angioma or 0,
        asterixis=note_in.asterixis or 0,
        follow_up_date=note_in.follow_up_date,
    )

    db.add(new_note)
    db.commit()
    db.refresh(new_note)

    log_audit(
        db,
        user_id=current_user.id,
        action="CREATE_CLINICAL_NOTE",
        resource="clinical_notes",
        resource_id=new_note.id,
        details={"patient_id": patient.id, "patient_code": patient.patient_id}
    )

    return {
        "success": True,
        "note": _format_clinical_note(new_note),
        "message": "Clinical note saved successfully"
    }


@app.get("/clinical-notes/{note_id}")
async def get_clinical_note_by_id(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    note = db.query(ClinicalNote).filter(ClinicalNote.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Clinical note not found")

    if not _is_admin(current_user) and note.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    return {
        "success": True,
        "note": _format_clinical_note(note)
    }


@app.put("/clinical-notes/{note_id}")
async def update_clinical_note(
    note_id: int,
    note_in: ClinicalNoteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    note = db.query(ClinicalNote).filter(ClinicalNote.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Clinical note not found")

    if not _is_admin(current_user) and note.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    update_dict = note_in.dict(exclude_unset=True)
    for field, val in update_dict.items():
        setattr(note, field, val)

    db.commit()
    db.refresh(note)

    return {
        "success": True,
        "note": _format_clinical_note(note),
        "message": "Clinical note updated successfully"
    }


@app.delete("/clinical-notes/{note_id}")
async def delete_clinical_note(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    note = db.query(ClinicalNote).filter(ClinicalNote.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Clinical note not found")

    if not _is_admin(current_user) and note.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    db.delete(note)
    db.commit()
    return {"success": True, "message": "Clinical note deleted"}


class ChronicConditionsUpdate(BaseModel):
    conditions: List[Dict[str, Any]]


@app.get("/patients/{patient_id}/chronic-conditions")
async def get_patient_chronic_conditions(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        pid = int(patient_id)
        patient = db.query(Patient).filter(Patient.id == pid).first()
    except ValueError:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    conditions = []
    if patient.chronic_conditions:
        try:
            conditions = json.loads(patient.chronic_conditions)
        except Exception:
            conditions = [{"name": patient.chronic_conditions, "status": "active"}]

    return {"success": True, "conditions": conditions}


@app.put("/patients/{patient_id}/chronic-conditions")
async def update_patient_chronic_conditions(
    patient_id: str,
    body: ChronicConditionsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        pid = int(patient_id)
        patient = db.query(Patient).filter(Patient.id == pid).first()
    except ValueError:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if not _is_admin(current_user) and patient.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your patient")

    patient.chronic_conditions = json.dumps(body.conditions, ensure_ascii=False)
    db.commit()
    db.refresh(patient)

    return {
        "success": True,
        "conditions": body.conditions,
        "message": "Chronic conditions updated successfully"
    }


# ─────────────────────────────────────────────────────────
# Phase 5: Liver Ultrasound, FibroScan & Imaging Endpoints
# ─────────────────────────────────────────────────────────

class UltrasoundExamCreate(BaseModel):
    patient_id: int
    exam_date: Optional[str] = None
    exam_type: Optional[str] = "ultrasound"  # ultrasound, fibroscan, ct, mri
    liver_size: Optional[str] = "Normal"     # Normal, Hepatomegaly, Shrunken / Atrophic
    echogenicity: Optional[str] = "Normal"   # Normal, Grade I Mild Fatty, Grade II Moderate Fatty, Grade III Severe Fatty, Coarse / Cirrhotic
    surface_contour: Optional[str] = "Smooth" # Smooth, Irregular / Nodular
    portal_vein_mm: Optional[float] = None
    portal_flow: Optional[str] = "Normal"
    spleen_size_cm: Optional[float] = None
    ascites: Optional[str] = "None"          # None, Mild, Moderate, Severe
    focal_lesion: Optional[str] = "None"     # None, Cyst, Hemangioma, Suspicious HCC, Multiple Nodules
    focal_lesion_desc: Optional[str] = None
    gallbladder: Optional[str] = "Normal"    # Normal, Stones, Sludge, Thickened Wall, Removed
    cbd_diameter_mm: Optional[float] = None
    fibroscan_kpa: Optional[float] = None
    fibroscan_cap: Optional[float] = None
    fibrosis_stage: Optional[str] = None
    steatosis_grade: Optional[str] = None
    impression: Optional[str] = None
    recommendations: Optional[str] = None
    image_urls: Optional[List[str]] = None


class UltrasoundExamUpdate(BaseModel):
    exam_date: Optional[str] = None
    exam_type: Optional[str] = None
    liver_size: Optional[str] = None
    echogenicity: Optional[str] = None
    surface_contour: Optional[str] = None
    portal_vein_mm: Optional[float] = None
    portal_flow: Optional[str] = None
    spleen_size_cm: Optional[float] = None
    ascites: Optional[str] = None
    focal_lesion: Optional[str] = None
    focal_lesion_desc: Optional[str] = None
    gallbladder: Optional[str] = None
    cbd_diameter_mm: Optional[float] = None
    fibroscan_kpa: Optional[float] = None
    fibroscan_cap: Optional[float] = None
    fibrosis_stage: Optional[str] = None
    steatosis_grade: Optional[str] = None
    impression: Optional[str] = None
    recommendations: Optional[str] = None
    image_urls: Optional[List[str]] = None


def _format_ultrasound_exam(exam: UltrasoundExam) -> dict:
    patient = exam.patient
    doctor = exam.doctor
    images = []
    if exam.image_urls:
        try:
            images = json.loads(exam.image_urls)
        except Exception:
            images = [exam.image_urls]
    return {
        "id": exam.id,
        "patient_id": exam.patient_id,
        "patient_name": patient.name if patient else "Unknown",
        "patient_code": patient.patient_id if patient else "",
        "doctor_id": exam.doctor_id,
        "doctor_name": doctor.full_name if doctor and doctor.full_name else (doctor.username if doctor else "Attending Physician"),
        "exam_date": exam.exam_date.isoformat() if exam.exam_date else None,
        "exam_type": exam.exam_type or "ultrasound",
        "liver_size": exam.liver_size or "Normal",
        "echogenicity": exam.echogenicity or "Normal",
        "surface_contour": exam.surface_contour or "Smooth",
        "portal_vein_mm": exam.portal_vein_mm,
        "portal_flow": exam.portal_flow or "Normal",
        "spleen_size_cm": exam.spleen_size_cm,
        "ascites": exam.ascites or "None",
        "focal_lesion": exam.focal_lesion or "None",
        "focal_lesion_desc": exam.focal_lesion_desc or "",
        "gallbladder": exam.gallbladder or "Normal",
        "cbd_diameter_mm": exam.cbd_diameter_mm,
        "fibroscan_kpa": exam.fibroscan_kpa,
        "fibroscan_cap": exam.fibroscan_cap,
        "fibrosis_stage": exam.fibrosis_stage or "",
        "steatosis_grade": exam.steatosis_grade or "",
        "impression": exam.impression or "",
        "recommendations": exam.recommendations or "",
        "image_urls": images,
        "created_at": exam.created_at.isoformat() if exam.created_at else None,
        "updated_at": exam.updated_at.isoformat() if exam.updated_at else None,
    }


@app.get("/ultrasounds")
async def get_ultrasound_exams(
    patient_id: Optional[Union[int, str]] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(UltrasoundExam)
    if not _is_admin(current_user):
        query = query.filter(UltrasoundExam.doctor_id == current_user.id)

    if patient_id is not None:
        try:
            pid_int = int(patient_id)
            query = query.filter(UltrasoundExam.patient_id == pid_int)
        except ValueError:
            p = db.query(Patient).filter(Patient.patient_id == str(patient_id)).first()
            if p:
                query = query.filter(UltrasoundExam.patient_id == p.id)
            else:
                return {"success": True, "exams": []}

    exams = query.order_by(desc(UltrasoundExam.created_at)).limit(limit).all()
    results = [_format_ultrasound_exam(e) for e in exams]
    return {"success": True, "exams": results}


@app.post("/ultrasounds")
async def create_ultrasound_exam(
    exam_in: UltrasoundExamCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(Patient.id == exam_in.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if not _is_admin(current_user) and patient.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to document exam for this patient")

    fibrosis_stage = exam_in.fibrosis_stage
    if exam_in.fibroscan_kpa is not None and not fibrosis_stage:
        kpa = exam_in.fibroscan_kpa
        if kpa < 7.0:
            fibrosis_stage = "F0-F1"
        elif kpa < 9.5:
            fibrosis_stage = "F2"
        elif kpa < 12.5:
            fibrosis_stage = "F3"
        else:
            fibrosis_stage = "F4 (Cirrhosis)"

    steatosis_grade = exam_in.steatosis_grade
    if exam_in.fibroscan_cap is not None and not steatosis_grade:
        cap = exam_in.fibroscan_cap
        if cap < 248:
            steatosis_grade = "S0 (Normal)"
        elif cap < 268:
            steatosis_grade = "S1 (Mild)"
        elif cap < 280:
            steatosis_grade = "S2 (Moderate)"
        else:
            steatosis_grade = "S3 (Severe)"

    img_json = json.dumps(exam_in.image_urls) if exam_in.image_urls else None

    exam_date = datetime.now()
    if exam_in.exam_date:
        try:
            exam_date = datetime.fromisoformat(exam_in.exam_date.replace("Z", "+00:00"))
        except Exception:
            pass

    new_exam = UltrasoundExam(
        patient_id=exam_in.patient_id,
        doctor_id=current_user.id,
        exam_date=exam_date,
        exam_type=exam_in.exam_type or "ultrasound",
        liver_size=exam_in.liver_size or "Normal",
        echogenicity=exam_in.echogenicity or "Normal",
        surface_contour=exam_in.surface_contour or "Smooth",
        portal_vein_mm=exam_in.portal_vein_mm,
        portal_flow=exam_in.portal_flow or "Normal",
        spleen_size_cm=exam_in.spleen_size_cm,
        ascites=exam_in.ascites or "None",
        focal_lesion=exam_in.focal_lesion or "None",
        focal_lesion_desc=exam_in.focal_lesion_desc,
        gallbladder=exam_in.gallbladder or "Normal",
        cbd_diameter_mm=exam_in.cbd_diameter_mm,
        fibroscan_kpa=exam_in.fibroscan_kpa,
        fibroscan_cap=exam_in.fibroscan_cap,
        fibrosis_stage=fibrosis_stage,
        steatosis_grade=steatosis_grade,
        impression=exam_in.impression,
        recommendations=exam_in.recommendations,
        image_urls=img_json,
    )

    db.add(new_exam)
    db.commit()
    db.refresh(new_exam)

    return {
        "success": True,
        "message": "Ultrasound / imaging examination saved successfully",
        "exam": _format_ultrasound_exam(new_exam)
    }


@app.get("/ultrasounds/{exam_id}")
async def get_ultrasound_exam(
    exam_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    exam = db.query(UltrasoundExam).filter(UltrasoundExam.id == exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    if not _is_admin(current_user) and exam.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    return {"success": True, "exam": _format_ultrasound_exam(exam)}


@app.put("/ultrasounds/{exam_id}")
async def update_ultrasound_exam(
    exam_id: int,
    exam_in: UltrasoundExamUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    exam = db.query(UltrasoundExam).filter(UltrasoundExam.id == exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    if not _is_admin(current_user) and exam.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    data = exam_in.dict(exclude_unset=True)
    if "image_urls" in data and data["image_urls"] is not None:
        data["image_urls"] = json.dumps(data["image_urls"])

    for field, val in data.items():
        if val is not None:
            setattr(exam, field, val)

    db.commit()
    db.refresh(exam)
    return {"success": True, "message": "Exam updated successfully", "exam": _format_ultrasound_exam(exam)}


@app.delete("/ultrasounds/{exam_id}")
async def delete_ultrasound_exam(
    exam_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    exam = db.query(UltrasoundExam).filter(UltrasoundExam.id == exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    if not _is_admin(current_user) and exam.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    db.delete(exam)
    db.commit()
    return {"success": True, "message": "Exam deleted successfully"}


@app.post("/ultrasounds/upload")
async def upload_ultrasound_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".pdf", ".dcm"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: JPG, PNG, WEBP, PDF, DCM")

    unique_filename = f"{uuid.uuid4().hex[:12]}_{os.path.basename(file.filename)}"
    save_path = os.path.join(UPLOAD_DIR, "imaging", unique_filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "url": f"/uploads/imaging/{unique_filename}",
        "filename": file.filename
    }


# ─────────────────────────────────────────────────────────
# Phase 6: Clinic Billing & Daily Financial Endpoints
# ─────────────────────────────────────────────────────────

class BillingCreate(BaseModel):
    patient_id: int
    visit_date: Optional[str] = None
    visit_type: str = "new_consultation"
    fee_iqd: int = 25000
    discount_iqd: int = 0
    final_iqd: Optional[int] = None
    is_paid: int = 1
    payment_method: str = "cash"
    notes: Optional[str] = None

class BillingUpdate(BaseModel):
    visit_type: Optional[str] = None
    fee_iqd: Optional[int] = None
    discount_iqd: Optional[int] = None
    final_iqd: Optional[int] = None
    is_paid: Optional[int] = None
    payment_method: Optional[str] = None
    notes: Optional[str] = None

def _format_billing_record(b: ClinicBilling) -> dict:
    return {
        "id": b.id,
        "bill_number": b.bill_number,
        "patient_id": b.patient_id,
        "patient_name": b.patient_name,
        "patient_code": b.patient_code,
        "doctor_id": b.doctor_id,
        "doctor_name": b.doctor_name,
        "visit_date": b.visit_date.isoformat() if b.visit_date else None,
        "visit_type": b.visit_type,
        "fee_iqd": b.fee_iqd,
        "discount_iqd": b.discount_iqd,
        "final_iqd": b.final_iqd,
        "is_paid": bool(b.is_paid),
        "payment_method": b.payment_method,
        "notes": b.notes,
        "created_at": b.created_at.isoformat() if b.created_at else None,
        "updated_at": b.updated_at.isoformat() if b.updated_at else None,
    }

def _generate_bill_number(db: Session) -> str:
    current_year = datetime.now().year
    prefix = f"INV-{current_year}-"
    last = db.query(ClinicBilling).filter(ClinicBilling.bill_number.like(f"{prefix}%")).order_by(desc(ClinicBilling.id)).first()
    if last and last.bill_number:
        try:
            seq = int(last.bill_number.split("-")[-1]) + 1
        except Exception:
            seq = 1
    else:
        seq = 1
    return f"{prefix}{seq:04d}"

@app.get("/billing")
async def get_billing_records(
    patient_id: Optional[int] = None,
    date: Optional[str] = None, # YYYY-MM-DD
    start_date: Optional[str] = None, # YYYY-MM-DD
    end_date: Optional[str] = None, # YYYY-MM-DD
    is_paid: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(ClinicBilling)

    if not _is_admin(current_user):
        query = query.filter(ClinicBilling.doctor_id == current_user.id)

    if patient_id is not None:
        query = query.filter(ClinicBilling.patient_id == patient_id)

    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            query = query.filter(func.date(ClinicBilling.visit_date) == target_date)
        except Exception:
            pass
    elif start_date or end_date:
        if start_date:
            try:
                s_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                query = query.filter(func.date(ClinicBilling.visit_date) >= s_date)
            except Exception:
                pass
        if end_date:
            try:
                e_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                query = query.filter(func.date(ClinicBilling.visit_date) <= e_date)
            except Exception:
                pass

    if is_paid is not None:
        query = query.filter(ClinicBilling.is_paid == is_paid)

    records = query.order_by(desc(ClinicBilling.visit_date)).all()
    return [_format_billing_record(r) for r in records]

@app.get("/billing/summary")
async def get_billing_summary(
    date: Optional[str] = None, # YYYY-MM-DD
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    secretary_split: float = 20.0, # percentage e.g. 20%
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(ClinicBilling)

    if not _is_admin(current_user):
        query = query.filter(ClinicBilling.doctor_id == current_user.id)

    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            query = query.filter(func.date(ClinicBilling.visit_date) == target_date)
        except Exception:
            pass
    elif start_date or end_date:
        if start_date:
            try:
                s_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                query = query.filter(func.date(ClinicBilling.visit_date) >= s_date)
            except Exception:
                pass
        if end_date:
            try:
                e_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                query = query.filter(func.date(ClinicBilling.visit_date) <= e_date)
            except Exception:
                pass

    records = query.all()

    total_bills = len(records)
    total_gross = sum(r.fee_iqd for r in records)
    total_discount = sum(r.discount_iqd for r in records)
    total_collected = sum(r.final_iqd for r in records if r.is_paid)
    pending_amount = sum(r.final_iqd for r in records if not r.is_paid)
    paid_count = sum(1 for r in records if r.is_paid)
    pending_count = sum(1 for r in records if not r.is_paid)

    # Breakdown by visit_type
    types_breakdown = {}
    for r in records:
        vt = r.visit_type or "other"
        if vt not in types_breakdown:
            types_breakdown[vt] = {"count": 0, "total_iqd": 0}
        types_breakdown[vt]["count"] += 1
        types_breakdown[vt]["total_iqd"] += (r.final_iqd if r.is_paid else 0)

    # Breakdown by payment method
    methods_breakdown = {}
    for r in records:
        pm = r.payment_method or "cash"
        if pm not in methods_breakdown:
            methods_breakdown[pm] = {"count": 0, "total_iqd": 0}
        methods_breakdown[pm]["count"] += 1
        methods_breakdown[pm]["total_iqd"] += (r.final_iqd if r.is_paid else 0)

    # Secretary and Doctor Split calculations
    secretary_cut = int(total_collected * (secretary_split / 100.0))
    doctor_net = total_collected - secretary_cut

    return {
        "total_bills": total_bills,
        "paid_count": paid_count,
        "pending_count": pending_count,
        "total_gross_iqd": total_gross,
        "total_discount_iqd": total_discount,
        "total_collected_iqd": total_collected,
        "pending_amount_iqd": pending_amount,
        "secretary_split_percent": secretary_split,
        "secretary_share_iqd": secretary_cut,
        "doctor_net_iqd": doctor_net,
        "by_visit_type": types_breakdown,
        "by_payment_method": methods_breakdown,
    }

@app.post("/billing")
async def create_billing_record(
    b_in: BillingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(Patient.id == b_in.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    bill_num = _generate_bill_number(db)
    
    calculated_final = b_in.final_iqd if b_in.final_iqd is not None else max(0, b_in.fee_iqd - b_in.discount_iqd)

    if b_in.visit_date:
        try:
            v_date = datetime.fromisoformat(b_in.visit_date.replace("Z", "+00:00"))
        except Exception:
            try:
                v_date = datetime.strptime(b_in.visit_date, "%Y-%m-%d")
            except Exception:
                v_date = datetime.now()
    else:
        v_date = datetime.now()

    new_bill = ClinicBilling(
        bill_number=bill_num,
        patient_id=patient.id,
        patient_name=patient.name,
        patient_code=patient.patient_id,
        doctor_id=current_user.id,
        doctor_name=current_user.full_name or current_user.username,
        visit_date=v_date,
        visit_type=b_in.visit_type,
        fee_iqd=b_in.fee_iqd,
        discount_iqd=b_in.discount_iqd,
        final_iqd=calculated_final,
        is_paid=b_in.is_paid,
        payment_method=b_in.payment_method,
        notes=b_in.notes
    )

    db.add(new_bill)
    db.commit()
    db.refresh(new_bill)

    return {"success": True, "message": "Bill created successfully", "bill": _format_billing_record(new_bill)}

@app.put("/billing/{bill_id}")
async def update_billing_record(
    bill_id: int,
    b_in: BillingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    bill = db.query(ClinicBilling).filter(ClinicBilling.id == bill_id).first()
    if not bill:
        raise HTTPException(status_code=404, detail="Bill record not found")

    if not _is_admin(current_user) and bill.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    data = b_in.dict(exclude_unset=True)
    for field, val in data.items():
        if val is not None:
            setattr(bill, field, val)

    if "final_iqd" not in data and ("fee_iqd" in data or "discount_iqd" in data):
        bill.final_iqd = max(0, bill.fee_iqd - bill.discount_iqd)

    db.commit()
    db.refresh(bill)
    return {"success": True, "message": "Bill updated successfully", "bill": _format_billing_record(bill)}

@app.delete("/billing/{bill_id}")
async def delete_billing_record(
    bill_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    bill = db.query(ClinicBilling).filter(ClinicBilling.id == bill_id).first()
    if not bill:
        raise HTTPException(status_code=404, detail="Bill record not found")

    if not _is_admin(current_user) and bill.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    db.delete(bill)
    db.commit()
    return {"success": True, "message": "Bill deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)