"""
Seed file: Populates the database with essential medications common in Iraqi pharmacies,
especially focused on Gastroenterology, Hepatology, and Internal Medicine.
All fields in English as required by the clinical design system.
"""

import json
from database import SessionLocal, engine, Base
from models import Medication

def seed_medications():
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    medications_data = [
        # ── Hepatoprotective & Bile Acid ──
        {
            "trade_name": "Ursofalk",
            "generic_name": "Ursodeoxycholic Acid",
            "category": "Hepatoprotective",
            "dosage_forms": json.dumps(["250mg capsule", "500mg tablet", "250mg/5ml suspension"]),
            "default_dose": "250mg",
            "default_freq": "BID (Twice daily)",
            "timing": "after_meal",
            "notes_ar": "Take with water after meals. Hepatoprotective, aids bile acid metabolism and dissolves cholesterol gallstones.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Legalon (Silymarin)",
            "generic_name": "Milk Thistle Extract / Silymarin",
            "category": "Hepatoprotective",
            "dosage_forms": json.dumps(["70mg capsule", "140mg capsule"]),
            "default_dose": "140mg",
            "default_freq": "TDS (Three times daily)",
            "timing": "after_meal",
            "notes_ar": "Natural antioxidant extract to protect and regenerate hepatocytes. Take after meals.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Hepamerz",
            "generic_name": "L-Ornithine L-Aspartate (LOLA)",
            "category": "Hepatoprotective",
            "dosage_forms": json.dumps(["3g sachet", "5g ampoule (IV)"]),
            "default_dose": "3g sachet",
            "default_freq": "TDS (Three times daily)",
            "timing": "after_meal",
            "notes_ar": "Dissolve sachet in water after meals to activate the urea cycle and reduce serum ammonia.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Livolin Forte",
            "generic_name": "Essential Phospholipids + Vitamins",
            "category": "Hepatoprotective",
            "dosage_forms": json.dumps(["Capsule"]),
            "default_dose": "1 capsule",
            "default_freq": "TDS (Three times daily)",
            "timing": "with_meal",
            "notes_ar": "Essential phospholipids and vitamins for hepatocyte membrane repair. Take with meals.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },

        # ── Diuretics (Ascites & Edema) ──
        {
            "trade_name": "Aldactone (Spironolactone)",
            "generic_name": "Spironolactone",
            "category": "Diuretics",
            "dosage_forms": json.dumps(["25mg tablet", "50mg tablet", "100mg tablet"]),
            "default_dose": "100mg",
            "default_freq": "QD (Once daily in morning)",
            "timing": "after_meal",
            "notes_ar": "Potassium-sparing diuretic, first-line therapy for hepatic ascites. Take in the morning with low-sodium diet.",
            "liver_warning": "Monitor potassium and renal function periodically to avoid hyperkalemia and hepatorenal syndrome.",
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Lasix",
            "generic_name": "Furosemide",
            "category": "Diuretics",
            "dosage_forms": json.dumps(["40mg tablet", "20mg/2ml ampoule"]),
            "default_dose": "40mg",
            "default_freq": "QD (Once daily in morning)",
            "timing": "after_meal",
            "notes_ar": "Loop diuretic combined with Spironolactone (ratio 40:100) for ascites control. Take in morning.",
            "liver_warning": "Avoid aggressive over-diuresis which may precipitate hypotension or hepatic encephalopathy.",
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },

        # ── Hepatic Encephalopathy & Bowel Regulators ──
        {
            "trade_name": "Duphalac (Lactulose)",
            "generic_name": "Lactulose Syrup",
            "category": "Hepatic Encephalopathy",
            "dosage_forms": json.dumps(["66.7g/100ml syrup (200ml/300ml)"]),
            "default_dose": "20ml - 30ml",
            "default_freq": "TDS (2-3 times daily)",
            "timing": "after_meal",
            "notes_ar": "Osmotic laxative and ammonia trap. Titrate dose to achieve 2-3 soft bowel movements daily.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Normix (Rifaximin)",
            "generic_name": "Rifaximin",
            "category": "Hepatic Encephalopathy",
            "dosage_forms": json.dumps(["200mg tablet", "550mg tablet"]),
            "default_dose": "550mg",
            "default_freq": "BID (Twice daily)",
            "timing": "after_meal",
            "notes_ar": "Non-absorbable antibiotic that reduces ammonia-producing gut bacteria for hepatic encephalopathy prophylaxis.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },

        # ── Viral Hepatitis Antivirals (HCV & HBV) ──
        {
            "trade_name": "Sovaldi / Sofosbuvir",
            "generic_name": "Sofosbuvir",
            "category": "Antivirals",
            "dosage_forms": json.dumps(["400mg tablet"]),
            "default_dose": "400mg",
            "default_freq": "QD (Once daily at fixed time)",
            "timing": "with_meal",
            "notes_ar": "Core direct-acting antiviral for Hepatitis C (HCV). 12-week regimen combined with Daclatasvir.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Daklinza / Daclatasvir",
            "generic_name": "Daclatasvir",
            "category": "Antivirals",
            "dosage_forms": json.dumps(["60mg tablet"]),
            "default_dose": "60mg",
            "default_freq": "QD (Once daily)",
            "timing": "with_meal",
            "notes_ar": "Taken concomitantly with Sofosbuvir for 12 weeks for chronic Hepatitis C.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Baraclude (Entecavir)",
            "generic_name": "Entecavir",
            "category": "Antivirals",
            "dosage_forms": json.dumps(["0.5mg tablet", "1mg tablet"]),
            "default_dose": "0.5mg",
            "default_freq": "QD (Once daily on empty stomach)",
            "timing": "before_meal",
            "notes_ar": "Antiviral for chronic Hepatitis B (HBV). Take on an empty stomach at least 2 hours before or after meals.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Viread (Tenofovir Disoproxil)",
            "generic_name": "Tenofovir Disoproxil Fumarate",
            "category": "Antivirals",
            "dosage_forms": json.dumps(["300mg tablet"]),
            "default_dose": "300mg",
            "default_freq": "QD (Once daily)",
            "timing": "with_meal",
            "notes_ar": "Antiviral for chronic Hepatitis B. Monitor serum creatinine, phosphorus, and renal function regularly.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },

        # ── Portal Hypertension & Variceal Bleeding Prophylaxis ──
        {
            "trade_name": "Inderal (Propranolol)",
            "generic_name": "Propranolol Hydrochloride",
            "category": "Cardiovascular / Portal HTN",
            "dosage_forms": json.dumps(["10mg tablet", "40mg tablet"]),
            "default_dose": "20mg - 40mg",
            "default_freq": "BID (Twice daily)",
            "timing": "before_meal",
            "notes_ar": "Non-selective beta-blocker to reduce portal pressure and prevent esophageal variceal hemorrhage (target heart rate: 55-60 bpm).",
            "liver_warning": "Contraindicated in severe asthma, severe bradycardia, or acute decompensated heart failure.",
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },

        # ── Gastroprotection (PPIs) ──
        {
            "trade_name": "Controloc / Pantodar (Pantoprazole)",
            "generic_name": "Pantoprazole",
            "category": "Gastroenterology",
            "dosage_forms": json.dumps(["20mg tablet", "40mg tablet", "40mg IV vial"]),
            "default_dose": "40mg",
            "default_freq": "QD (Once daily in morning)",
            "timing": "before_meal",
            "notes_ar": "Proton pump inhibitor for gastroprotection and acid suppression. Take 30 minutes before breakfast.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Nexium (Esomeprazole)",
            "generic_name": "Esomeprazole",
            "category": "Gastroenterology",
            "dosage_forms": json.dumps(["20mg tablet", "40mg tablet", "40mg IV vial"]),
            "default_dose": "40mg",
            "default_freq": "QD (Once daily in morning)",
            "timing": "before_meal",
            "notes_ar": "PPI for gastric protection, duodenal ulcer, and reflux. Take 30 minutes before breakfast.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },

        # ── Analgesics & Cautionary Meds ──
        {
            "trade_name": "Panadol (Paracetamol)",
            "generic_name": "Paracetamol / Acetaminophen",
            "category": "Analgesics",
            "dosage_forms": json.dumps(["500mg tablet", "1000mg effervescent"]),
            "default_dose": "500mg",
            "default_freq": "PRN (As needed)",
            "timing": "after_meal",
            "notes_ar": "Relatively safe analgesic. Strict limit: maximum 2g (4 x 500mg tablets) in 24 hours for cirrhosis patients.",
            "liver_warning": "Liver Safety Warning: Maximum 2g daily in cirrhosis. Strict avoidance of NSAIDs (Ibuprofen, Diclofenac).",
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },

        # ── Vitamins & Supplements ──
        {
            "trade_name": "Konakion (Vitamin K1)",
            "generic_name": "Phytomenadione (Vitamin K1)",
            "category": "Vitamins / Hematology",
            "dosage_forms": json.dumps(["10mg ampoule (Oral / IM / IV)"]),
            "default_dose": "10mg",
            "default_freq": "QD (Once daily for 3 days)",
            "timing": "after_meal",
            "notes_ar": "Improves prothrombin time (PT/INR) in cholestasis and malabsorption syndromes.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Neurobion (B1+B6+B12)",
            "generic_name": "Vitamin B-Complex",
            "category": "Vitamins",
            "dosage_forms": json.dumps(["Tablet", "Ampoule (IM)"]),
            "default_dose": "1 tablet",
            "default_freq": "QD (Once daily)",
            "timing": "after_meal",
            "notes_ar": "B-complex vitamins to support neuro-metabolic function and general cellular vitality.",
            "liver_warning": None,
            "is_liver_safe": 1,
            "common_in_iraq": 1
        },
        {
            "trade_name": "Metformin (Glucophage)",
            "generic_name": "Metformin Hydrochloride",
            "category": "Endocrinology / Diabetes",
            "dosage_forms": json.dumps(["500mg tablet", "850mg tablet", "1000mg XR"]),
            "default_dose": "500mg - 1000mg",
            "default_freq": "BID (Twice daily)",
            "timing": "after_meal",
            "notes_ar": "Improves insulin sensitivity in NAFLD / NASH patients with diabetes. Take with or after meals.",
            "liver_warning": "Discontinue immediately in severe hepatic failure, acute illness, or advanced renal impairment.",
            "is_liver_safe": 1,
            "common_in_iraq": 1
        }
    ]

    for med in medications_data:
        existing = db.query(Medication).filter(Medication.trade_name == med["trade_name"]).first()
        if existing:
            existing.generic_name = med["generic_name"]
            existing.category = med["category"]
            existing.dosage_forms = med["dosage_forms"]
            existing.default_dose = med["default_dose"]
            existing.default_freq = med["default_freq"]
            existing.timing = med["timing"]
            existing.notes_ar = med["notes_ar"]
            existing.liver_warning = med["liver_warning"]
            existing.is_liver_safe = med["is_liver_safe"]
            existing.common_in_iraq = med["common_in_iraq"]
        else:
            new_med = Medication(**med)
            db.add(new_med)

    db.commit()
    print("Successfully synced all medications in English.")
    db.close()

if __name__ == "__main__":
    seed_medications()
