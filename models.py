import json as _json
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

# ─────────────────────────────────────────────────────────
# Permission Defaults
# ─────────────────────────────────────────────────────────
DEFAULT_PERMISSIONS = {
    "can_view_dashboard":  True,
    "can_run_analysis":    False,
    "can_use_chatbot":     True,
    "can_view_reports":    False,
    "can_view_patients":   True,
    "can_create_patients": False,
    "can_edit_patients":   False,
    "can_delete_patients": False,
    "can_view_records":    False,
    "can_manage_prescriptions": False,
    "can_manage_clinical_notes": False,
    "can_manage_billing":  False,
    "can_manage_users":    False,
    "can_view_audit_logs": False,
    "can_access_admin":    False,
}

ALL_PERMISSIONS = {k: True for k in DEFAULT_PERMISSIONS}

DOCTOR_PRESET_PERMISSIONS = {
    "can_view_dashboard":  True,
    "can_run_analysis":    True,
    "can_use_chatbot":     True,
    "can_view_reports":    True,
    "can_view_patients":   True,
    "can_create_patients": True,
    "can_edit_patients":   True,
    "can_delete_patients": False,
    "can_view_records":    True,
    "can_manage_prescriptions": True,
    "can_manage_clinical_notes": True,
    "can_manage_billing":  True,
    "can_manage_users":    False,
    "can_view_audit_logs": False,
    "can_access_admin":    False,
}

# ─────────────────────────────────────────────────────────
# Existing Models (UNCHANGED)
# ─────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    patient_id = Column(String(50), unique=True, nullable=False)
    birth_date = Column(String(10), nullable=True)  # Store as string YYYY-MM-DD
    email = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    profile_picture = Column(String(500), nullable=True)  # URL or path to profile picture
    department = Column(String(100), nullable=True)  # Medical department
    doctor_name = Column(String(255), nullable=True)  # Attending physician/supervisor
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Owning doctor
    status = Column(String(20), nullable=False, default="active")  # active, archived
    chronic_conditions = Column(Text, nullable=True)  # JSON array of chronic illnesses
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    doctor = relationship("User", foreign_keys=[doctor_id], back_populates="patients")
    lab_tests = relationship("LabTest", back_populates="patient", cascade="all, delete-orphan")
    medical_reports = relationship("MedicalReport", back_populates="patient", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="patient", cascade="all, delete-orphan")
    clinical_notes = relationship("ClinicalNote", back_populates="patient", cascade="all, delete-orphan")
    ultrasound_exams = relationship("UltrasoundExam", back_populates="patient", cascade="all, delete-orphan")
    billing_records = relationship("ClinicBilling", back_populates="patient", cascade="all, delete-orphan")

class LabTest(Base):
    __tablename__ = "lab_tests"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    test_name = Column(String(255), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=False)
    normal_range = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # normal, high, low, critical
    date = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="lab_tests")

class MedicalReport(Base):
    __tablename__ = "medical_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Owning doctor
    diagnosis = Column(String(500), nullable=False)
    confidence = Column(Float, nullable=False)  # 0-100
    advice = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="active")  # active, archived, finalized
    is_finalized = Column(Integer, nullable=False, default=0)  # 0=false, 1=true
    risk_level = Column(String(20), nullable=True)  # high, medium, low
    detailed_results = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    doctor = relationship("User", foreign_keys=[doctor_id], back_populates="medical_reports")
    patient = relationship("Patient", back_populates="medical_reports")

# ─────────────────────────────────────────────────────────
# E-Prescription Models (المرحلة 1: نظام الوصفات الطبية)
# ─────────────────────────────────────────────────────────

class Medication(Base):
    __tablename__ = "medications"

    id             = Column(Integer, primary_key=True, index=True)
    trade_name     = Column(String(255), nullable=False, index=True)
    generic_name   = Column(String(255), nullable=False, index=True)
    category       = Column(String(100), nullable=True)  # e.g., Hepatology, Diuretics, Antivirals
    dosage_forms   = Column(Text, nullable=True)         # JSON list e.g. ["250mg capsule", "500mg tablet"]
    default_dose   = Column(String(100), nullable=True)
    default_freq   = Column(String(100), nullable=True)  # e.g., TDS, BID, QD
    timing         = Column(String(100), nullable=True)  # after_meal, before_meal, bedtime
    notes_ar       = Column(Text, nullable=True)         # تعليمات الاستعمال بالعربية
    liver_warning  = Column(Text, nullable=True)         # تحذير لمرضى التليف/القصور الكبدي
    is_liver_safe  = Column(Integer, default=1)          # 1=آمن، 0=يحتاج حذر أو ممنوع في الفشل الكبدي
    common_in_iraq = Column(Integer, default=1)          # 1=شائع في الصيدليات العراقية
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

class Prescription(Base):
    __tablename__ = "prescriptions"

    id                  = Column(Integer, primary_key=True, index=True)
    prescription_number = Column(String(50), unique=True, nullable=False, index=True)
    patient_id          = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id           = Column(Integer, ForeignKey("users.id"), nullable=True)
    diagnosis           = Column(String(500), nullable=True)
    notes               = Column(Text, nullable=True)
    follow_up_date      = Column(String(100), nullable=True)
    status              = Column(String(20), nullable=False, default="active")  # active, completed, cancelled
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    updated_at          = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="prescriptions")
    doctor  = relationship("User", foreign_keys=[doctor_id], back_populates="prescriptions")
    items   = relationship("PrescriptionItem", back_populates="prescription", cascade="all, delete-orphan")

class PrescriptionItem(Base):
    __tablename__ = "prescription_items"

    id              = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=False)
    medication_id   = Column(Integer, ForeignKey("medications.id"), nullable=True)
    medication_name = Column(String(255), nullable=False)
    generic_name    = Column(String(255), nullable=True)
    dose            = Column(String(100), nullable=False)
    frequency       = Column(String(100), nullable=False)
    timing          = Column(String(100), nullable=True)
    duration        = Column(String(100), nullable=True)
    instructions_ar = Column(Text, nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    prescription = relationship("Prescription", back_populates="items")
    medication   = relationship("Medication")

# ─────────────────────────────────────────────────────────
# Phase 3: Clinical Notes (SOAP) & Follow-up Tracking
# ─────────────────────────────────────────────────────────

class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id                  = Column(Integer, primary_key=True, index=True)
    patient_id          = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id           = Column(Integer, ForeignKey("users.id"), nullable=True)
    visit_date          = Column(DateTime(timezone=True), server_default=func.now())
    visit_type          = Column(String(50), nullable=False, default="follow_up") # new, follow_up, routine

    # SOAP components
    subjective          = Column(Text, nullable=True) # Chief complaint & history
    objective           = Column(Text, nullable=True) # Physical examination findings
    assessment          = Column(Text, nullable=True) # Clinical assessment & diagnosis
    plan                = Column(Text, nullable=True) # Diagnostic & therapeutic plan

    # Vital Signs
    blood_pressure      = Column(String(50), nullable=True) # e.g. "120/80"
    heart_rate          = Column(Integer, nullable=True)    # bpm
    weight              = Column(Float, nullable=True)      # kg
    temperature         = Column(Float, nullable=True)      # °C

    # Liver-Specific Physical Signs
    jaundice            = Column(String(50), nullable=True) # None, Mild, Moderate, Severe
    ascites             = Column(String(50), nullable=True) # None, Mild, Moderate, Tense
    edema               = Column(String(50), nullable=True) # None, Mild (+1), Moderate (+2), Severe (+3)
    hepatomegaly        = Column(Integer, default=0)        # 0=No, 1=Yes
    splenomegaly        = Column(Integer, default=0)        # 0=No, 1=Yes
    spider_angioma      = Column(Integer, default=0)        # 0=No, 1=Yes
    asterixis           = Column(Integer, default=0)        # 0=No, 1=Yes (Encephalopathy flap)

    follow_up_date      = Column(String(100), nullable=True)
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    updated_at          = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="clinical_notes")
    doctor  = relationship("User", foreign_keys=[doctor_id], back_populates="clinical_notes")

# ─────────────────────────────────────────────────────────
# Phase 5: Liver Ultrasound, Imaging & FibroScan Documentation
# ─────────────────────────────────────────────────────────

class UltrasoundExam(Base):
    __tablename__ = "ultrasound_exams"

    id                  = Column(Integer, primary_key=True, index=True)
    patient_id          = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id           = Column(Integer, ForeignKey("users.id"), nullable=True)
    exam_date           = Column(DateTime(timezone=True), server_default=func.now())
    exam_type           = Column(String(50), nullable=False, default="ultrasound") # ultrasound, fibroscan, ct, mri

    # Liver Parenchyma & Dimensions
    liver_size          = Column(String(50), nullable=True, default="Normal") # Normal, Hepatomegaly, Shrunken / Atrophic
    echogenicity        = Column(String(50), nullable=True, default="Normal") # Normal, Grade I Mild Fatty, Grade II Moderate Fatty, Grade III Severe Fatty, Coarse / Cirrhotic
    surface_contour     = Column(String(50), nullable=True, default="Smooth") # Smooth, Irregular / Nodular

    # Vascular & Portal System
    portal_vein_mm      = Column(Float, nullable=True) # mm, normal < 13
    portal_flow         = Column(String(50), nullable=True, default="Normal") # Normal, Slowed, Hepatofugal, Thrombosis
    spleen_size_cm      = Column(Float, nullable=True) # cm, normal < 12-13

    # Complications & Lesions
    ascites             = Column(String(50), nullable=True, default="None") # None, Mild, Moderate, Severe
    focal_lesion        = Column(String(50), nullable=True, default="None") # None, Cyst, Hemangioma, Suspicious HCC, Multiple Nodules
    focal_lesion_desc   = Column(Text, nullable=True)

    # Biliary Tree
    gallbladder         = Column(String(50), nullable=True, default="Normal") # Normal, Stones, Sludge, Thickened Wall, Removed
    cbd_diameter_mm     = Column(Float, nullable=True) # mm, normal < 6-7

    # FibroScan / Elastography (optional)
    fibroscan_kpa       = Column(Float, nullable=True) # Liver stiffness in kPa
    fibroscan_cap       = Column(Float, nullable=True) # Controlled Attenuation Parameter in dB/m
    fibrosis_stage      = Column(String(20), nullable=True) # F0, F1, F2, F3, F4
    steatosis_grade     = Column(String(20), nullable=True) # S0, S1, S2, S3

    # Summary & Recommendations
    impression          = Column(Text, nullable=True)
    recommendations     = Column(Text, nullable=True)
    image_urls          = Column(Text, nullable=True) # JSON array of image URLs/paths

    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    updated_at          = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="ultrasound_exams")
    doctor  = relationship("User", foreign_keys=[doctor_id], back_populates="ultrasound_exams")

# ─────────────────────────────────────────────────────────
# Clinic Billing & Finance Models (المرحلة 6: إدارة الكشفيات والمالية)
# ─────────────────────────────────────────────────────────

class ClinicBilling(Base):
    __tablename__ = "clinic_billing"

    id             = Column(Integer, primary_key=True, index=True)
    bill_number    = Column(String(50), unique=True, nullable=False, index=True) # e.g. INV-2026-0001
    patient_id     = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id      = Column(Integer, ForeignKey("users.id"), nullable=True)
    doctor_name    = Column(String(255), nullable=True)
    patient_name   = Column(String(255), nullable=True)
    patient_code   = Column(String(50), nullable=True)

    visit_date     = Column(DateTime(timezone=True), nullable=False)
    visit_type     = Column(String(50), nullable=False, default="new_consultation")
    # Options: new_consultation, follow_up, ultrasound, fibroscan, procedure, free_exempt

    fee_iqd        = Column(Integer, nullable=False, default=25000)
    discount_iqd   = Column(Integer, nullable=False, default=0)
    final_iqd      = Column(Integer, nullable=False, default=25000)

    is_paid        = Column(Integer, nullable=False, default=1)  # 1=paid, 0=pending/unpaid
    payment_method = Column(String(50), nullable=False, default="cash") # cash, card, free, installment
    notes          = Column(Text, nullable=True)

    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    updated_at     = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="billing_records")
    doctor  = relationship("User", foreign_keys=[doctor_id], back_populates="billing_records")

# ─────────────────────────────────────────────────────────
# Auth Models
# ─────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(50), unique=True, nullable=False)
    email           = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name       = Column(String(255), nullable=True)
    role            = Column(String(20), nullable=False, default="user")  # label only
    is_active       = Column(Integer, nullable=False, default=1)  # 0=disabled, 1=active
    permissions     = Column(Text, nullable=False, default=_json.dumps(DEFAULT_PERMISSIONS))
    last_login      = Column(DateTime(timezone=True), nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patients = relationship("Patient", back_populates="doctor", foreign_keys="Patient.doctor_id")
    medical_reports = relationship("MedicalReport", back_populates="doctor", foreign_keys="MedicalReport.doctor_id")
    prescriptions = relationship("Prescription", back_populates="doctor", foreign_keys="Prescription.doctor_id")
    clinical_notes = relationship("ClinicalNote", back_populates="doctor", foreign_keys="ClinicalNote.doctor_id")
    ultrasound_exams = relationship("UltrasoundExam", back_populates="doctor", foreign_keys="UltrasoundExam.doctor_id")
    billing_records = relationship("ClinicBilling", back_populates="doctor", foreign_keys="ClinicBilling.doctor_id")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

    def get_permissions(self) -> dict:
        """Parse the JSON permissions column, merging with defaults for any missing keys."""
        try:
            perms = _json.loads(self.permissions) if self.permissions else {}
        except (ValueError, TypeError):
            perms = {}
        # Ensure all known keys exist (forward-compatible)
        merged = {**DEFAULT_PERMISSIONS, **perms}
        return merged

    def has_permission(self, perm: str) -> bool:
        """Check if the user has a specific permission."""
        return bool(self.get_permissions().get(perm, False))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    action      = Column(String(100), nullable=False)   # e.g. "LOGIN", "CREATE_PATIENT"
    resource    = Column(String(100), nullable=True)     # e.g. "patients"
    resource_id = Column(Integer, nullable=True)
    details     = Column(Text, nullable=True)            # JSON string with extra info
    ip_address  = Column(String(45), nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")