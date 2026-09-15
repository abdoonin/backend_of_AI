#!/usr/bin/env python3
"""
DiagnosisEngine: Multi-Stage Liver Disease Prediction System

Implements the complete backend logic for liver disease diagnosis using 6 XGBoost models:
- Gate model: Determines if patient is at risk
- Cancer model: Predicts cancer risk
- Fatty Liver model: Detects NAFLD
- Hepatitis models: 3-step flow (stage, complications, status)

Handles all data mapping, transformations, and generates medical advice.
"""

import os
import math
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosisEngine:
    """
    Smart Dispatcher: Routes data from master profile to appropriate ML models.
    Handles all feature extraction, encoding, and prediction logic.
    """

    # Required keys in user_profile (28 standardized keys)
    REQUIRED_KEYS = [
        'age', 'gender', 'bmi', 'smoking', 'alcohol', 'activity', 'cancer_history',
        'genetic_risk', 'ascites', 'hepatomegaly', 'spiders', 'edema', 'bilirubin',
        'albumin', 'alp', 'alt', 'ast', 'platelets', 'prothrombin', 'copper',
        'cholesterol', 'creatinine', 'glucose', 'ggt', 'triglycerides', 'uric_acid',
        'hdl', 'total_proteins'
    ]

    # Strict feature maps for each model (hard-coded for reliability)
    GATE_KEYS = ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_protiens', 'albumin', 'albumin_and_globulin_ratio']

    CANCER_KEYS = ['age', 'gender', 'bmi', 'smoking', 'genetic_risk', 'activity', 'alcohol', 'cancer_history']

    FATTY_KEYS = ['albumin', 'alp', 'ast', 'alt', 'cholesterol', 'creatinine', 'glucose', 'ggt', 'bilirubin', 'triglycerides', 'uric_acid', 'platelets', 'hdl']

    HEP_STAGE_KEYS = ['bilirubin', 'cholesterol', 'albumin', 'copper', 'alp', 'ast', 'triglycerides', 'platelets', 'prothrombin', 'age', 'gender', 'ascites', 'hepatomegaly', 'spiders', 'edema']

    HEP_COMP_KEYS = ['bilirubin', 'cholesterol', 'albumin', 'copper', 'alp', 'ast', 'triglycerides', 'platelets', 'prothrombin', 'age', 'gender', 'hepatomegaly', 'spiders', 'edema']  # 14 features, no ascites

    # ── HCV model contracts ────────────────────────────────────────────────
    #
    # Read off each .pkl's own `feature_names_in_`, not copied from the docs.
    # These are sklearn Pipelines with a ColumnTransformer, so they need a
    # NAMED DataFrame -- the previous models took positional arrays, and
    # passing one here raises "X has 15 features, but ColumnTransformer is
    # expecting 19".
    #
    # ORDER OF EXECUTION IS status -> stage, which is the REVERSE of what the
    # model documentation describes. It is forced by the artifacts: the stage
    # model requires `Status` as an input, and the status model has no `Stage`
    # among its 18 features. The old cascade ran stage -> status and injected
    # the stage at index 9; that is gone, because there is nowhere to inject it.
    HEPC_STAGE_COLS = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
                       'Tryglicerides', 'Platelets', 'Prothrombin', 'Status', 'Age', 'Sex',
                       'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI',
                       'Bilirubin_Albumin', 'Copper_Platelets']

    HEPC_STATUS_COLS = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
                        'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex', 'Ascites',
                        'Hepatomegaly', 'Spiders', 'Edema', 'APRI', 'ALBI_Score',
                        'Bili_Alb_Ratio']

    HEPC_COMP_COLS = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
                      'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex',
                      'Hepatomegaly', 'Spiders', 'Edema']

    def __init__(self, model_dir: str = None):
        """
        Initialize the DiagnosisEngine by loading all ML models.

        Args:
            model_dir: Directory containing the .pkl model files. Defaults to script directory.
        """
        if model_dir is None:
            model_dir = os.path.dirname(__file__)

        self.models = {}
        self._load_models(model_dir)

    def _load_models(self, model_dir: str) -> None:
        """Load all 6 XGBoost models from the specified directory."""
        model_files = {
            'gate': 'gate_model.pkl',
            'cancer': 'cancer_model.pkl',
            'fatty_liver': 'fatty_liver_model.pkl',
            # HCV models, swapped in 2026-08-10 on Ali's decision. The previous
            # five-class files (hepatitis_stage.pkl, hepatitis_complications.pkl,
            # hepatitis_status.pkl) are still on disk and untouched -- reverting
            # is these three strings and nothing else.
            'hepatitis_stage': 'hepatitisC_stage_model.pkl',
            'hepatitis_complications': 'hepatitisC_complications.pkl',
            'hepatitis_status': 'hepatitisC_status_model.pkl'
        }

        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            try:
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model successfully")
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
                raise RuntimeError(f"Cannot load {model_name} model: {e}")

    def predict_full_diagnosis(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method: Perform complete multi-stage liver disease diagnosis.

        Args:
            user_profile: Dictionary with 28 standardized keys containing patient data

        Returns:
            Dictionary with diagnosis results, confidence scores, and medical advice
        """
        # Validate input
        self._validate_user_profile(user_profile)

        # Convert string values to appropriate types
        processed_profile = self._preprocess_profile(user_profile)

        # Step 1: Gate Model - Check if patient is at risk
        gate_result = self._predict_gate(processed_profile)

        if gate_result['is_healthy']:  # prediction == 1 means healthy
            return self._generate_healthy_response(gate_result)

        # Gate detected sick patient - proceed to sub-models
        logger.info("Gate detected sick patient. Activating sub-models...")

        # Step 2: Parallel diagnostic branches
        results = {
            'gate': gate_result,
            'cancer': self._predict_cancer(processed_profile),
            'fatty_liver': self._predict_fatty_liver(processed_profile),
            'hepatitis': self._predict_hepatitis(processed_profile)
        }

        # Step 3: Generate final diagnosis and advice
        final_diagnosis = self._generate_final_diagnosis(results)

        return {
            'success': True,
            'diagnosis': final_diagnosis['diagnosis'],
            'confidence': final_diagnosis['confidence'],
            'advice': final_diagnosis['advice'],
            'detailed_results': results,
            'risk_level': final_diagnosis['risk_level']
        }

    def predict_gate_only(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform gate-only screening analysis.

        Args:
            user_profile: Dictionary with required keys for gate model

        Returns:
            Dictionary with gate prediction results
        """
        # Validate input
        self._validate_user_profile(user_profile)

        # Convert string values to appropriate types
        processed_profile = self._preprocess_profile(user_profile)

        # Gate Model prediction
        gate_result = self._predict_gate(processed_profile)

        if gate_result['is_healthy']:
            return self._generate_healthy_response(gate_result)
        else:
            # Sick patient detected by gate
            return {
                'success': True,
                'diagnosis': 'Potential Risk Detected',
                'confidence': 80,
                'advice': 'Gate screening indicates potential liver disease risk. Detailed analysis recommended for accurate diagnosis.',
                'detailed_results': {'gate': gate_result},
                'risk_level': 'moderate'
            }

    def predict_cancer_only(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cancer-only analysis.

        Args:
            user_profile: Dictionary with required keys for cancer model

        Returns:
            Dictionary with cancer analysis results
        """
        # Validate input - only check for CANCER_KEYS
        missing_keys = [key for key in self.CANCER_KEYS if key not in user_profile]
        if missing_keys:
            raise ValueError(f"Missing required keys for cancer analysis: {missing_keys}")

        # Convert string values to appropriate types
        processed_profile = self._preprocess_profile(user_profile)

        # Cancer Model prediction
        cancer_result = self._predict_cancer(processed_profile)

        return {
            'success': True,
            'analysis_type': 'cancer',
            'results': {'cancer': cancer_result}
        }

    def predict_fatty_liver_only(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform fatty liver-only analysis.

        Args:
            user_profile: Dictionary with required keys for fatty liver model

        Returns:
            Dictionary with fatty liver analysis results
        """
        # Validate input - only check for FATTY_KEYS
        missing_keys = [key for key in self.FATTY_KEYS if key not in user_profile]
        if missing_keys:
            raise ValueError(f"Missing required keys for fatty liver analysis: {missing_keys}")

        # Convert string values to appropriate types
        processed_profile = self._preprocess_profile(user_profile)

        # Fatty Liver Model prediction
        fatty_liver_result = self._predict_fatty_liver(processed_profile)

        return {
            'success': True,
            'analysis_type': 'fatty_liver',
            'results': {'fatty_liver': fatty_liver_result}
        }

    def predict_hepatitis_only(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hepatitis-only analysis.

        Args:
            user_profile: Dictionary with required keys for hepatitis models

        Returns:
            Dictionary with hepatitis analysis results
        """
        # Validate input - only check for HEP_STAGE_KEYS (most comprehensive set)
        missing_keys = [key for key in self.HEP_STAGE_KEYS if key not in user_profile]
        if missing_keys:
            raise ValueError(f"Missing required keys for hepatitis analysis: {missing_keys}")

        # Convert string values to appropriate types
        processed_profile = self._preprocess_profile(user_profile)

        # Hepatitis Model prediction
        hepatitis_result = self._predict_hepatitis(processed_profile)

        return {
            'success': True,
            'analysis_type': 'hepatitis',
            'results': {'hepatitis': hepatitis_result}
        }

    def _validate_user_profile(self, user_profile: Dict[str, Any]) -> None:
        """Validate that all required keys are present in user_profile."""
        # Get mode from request, default to 'full' for safety
        mode = user_profile.get('mode', 'full')

        if mode == 'gate':
            # Stage 1: The 10 Gate Keys ONLY
            required_keys = self.GATE_KEYS
        else:
            # Stage 2: The Full 28 Keys (Existing Logic)
            required_keys = self.REQUIRED_KEYS

        # Proceed with existing check...
        missing_keys = [key for key in required_keys if key not in user_profile]
        if missing_keys:
            raise ValueError(f"Missing required keys in user_profile: {missing_keys}")

    def _preprocess_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string inputs to appropriate numeric types where needed."""
        processed = user_profile.copy()

        # Convert numeric strings to floats
        numeric_keys = [
            'age', 'bmi', 'bilirubin', 'albumin', 'alp', 'alt', 'ast', 'platelets',
            'prothrombin', 'copper', 'cholesterol', 'creatinine', 'glucose', 'ggt',
            'triglycerides', 'uric_acid', 'hdl', 'total_proteins'
        ]

        for key in numeric_keys:
            if key in processed and isinstance(processed[key], str):
                try:
                    processed[key] = float(processed[key])
                except ValueError:
                    raise ValueError(f"Invalid numeric value for {key}: {processed[key]}")

        return processed

    def _extract_features(self, profile: Dict[str, Any], required_keys: List[str]) -> np.ndarray:
        """
        Extracts specific values from the 28-key 'profile' dictionary
        in the EXACT order defined by 'required_keys'.
        Handles encoding for categorical variables and derived features.
        """
        # Map gate keys from full names to short names if needed
        if required_keys == self.GATE_KEYS:
            profile = {
                'age': profile.get('age'),
                'gender': profile.get('gender'),
                'bilirubin': profile.get('bilirubin') or profile.get('total_bilirubin'),
                'bilirubin_direct': profile.get('bilirubin_direct') or profile.get('direct_bilirubin'),
                'alp': profile.get('alp') or profile.get('alkaline_phosphotase'),
                'alt': profile.get('alt') or profile.get('alamine_aminotransferase'),
                'ast': profile.get('ast') or profile.get('aspartate_aminotransferase'),
                'total_proteins': profile.get('total_proteins') or profile.get('total_protiens'),
                'albumin': profile.get('albumin'),
                'ag_ratio': profile.get('ag_ratio') or profile.get('albumin_and_globulin_ratio')
            }
            required_keys = ['age', 'gender', 'bilirubin', 'bilirubin_direct', 'alp', 'alt', 'ast', 'total_proteins', 'albumin', 'ag_ratio']

        feature_vector = []

        for key in required_keys:
            # Handle derived features first
            if key == 'bilirubin_direct':
                # Estimate direct bilirubin as 30% of total
                value = profile['bilirubin'] * 0.3
            elif key == 'ag_ratio':
                # Calculate A/G ratio
                total_proteins = profile.get('total_proteins', 7.0)
                albumin = profile.get('albumin', 4.0)
                value = albumin / (total_proteins - albumin) if total_proteins > albumin else 0
            elif key not in profile:
                raise ValueError(f"Missing required key: {key}")
            else:
                value = profile[key]

            # Handle categorical encoding
            if key == 'gender':
                # Different models have different gender encodings
                if required_keys == self.GATE_KEYS:
                    # Gate: Male=1, Female=0
                    value = 1 if str(value).lower() == 'male' else 0
                elif required_keys == self.CANCER_KEYS:
                    # Cancer: Male=0, Female=1 (opposite)
                    value = 0 if str(value).lower() == 'male' else 1
                else:
                    # Hepatitis models: Male=1, Female=0
                    value = 1 if str(value).lower() == 'male' else 0
            elif key in ['smoking', 'cancer_history', 'ascites', 'hepatomegaly', 'spiders']:
                # Binary encodings
                value = 1 if str(value).lower() in ['yes', 'y'] else 0
            elif key == 'genetic_risk':
                # Genetic risk levels
                value = {'low': 0, 'medium': 1, 'high': 2}.get(str(value).lower(), 0)
            elif key == 'edema':
                # Edema levels
                edema_str = str(value).lower()
                if edema_str == 'severe':
                    value = 1.0
                elif edema_str == 'slight':
                    value = 0.5
                else:  # 'no'
                    value = 0.0
            elif key in ['activity', 'alcohol']:
                # Default to moderate (1) for activity/alcohol
                value = 1

            feature_vector.append(value)

        # Return as 2D array for XGBoost (1 sample, N features)
        return np.array([feature_vector])

    def _predict_gate(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Gate model prediction using 10 features."""
        X_input = self._extract_features(profile, self.GATE_KEYS)
        prediction = int(self.models['gate'].predict(X_input)[0])

        # ADDITIVE, 2026-08-09. The gate returned no probability, so the UI
        # showed a hardcoded literal (80 sick / 95 healthy) as if it were a
        # model confidence. The model could always supply a real one.
        probabilities = self.models['gate'].predict_proba(X_input)[0]

        return {
            'prediction': prediction,  # 0 = Sick, 1 = Healthy
            'is_healthy': prediction == 1,  # Only 1 means healthy
            'features_used': self.GATE_KEYS,
            'confidence_pct': round(float(probabilities[prediction]) * 100, 1),
            'probability_sick': round(float(probabilities[0]) * 100, 1),
            'probability_healthy': round(float(probabilities[1]) * 100, 1)
        }

    def _predict_cancer(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Cancer risk prediction with 5-tier assessment."""
        X_input = self._extract_features(profile, self.CANCER_KEYS)
        risk_prob = float(self.models['cancer'].predict_proba(X_input)[0][1])
        risk_pct = risk_prob * 100

        # 5-tier risk assessment
        if risk_pct <= 10:
            risk_level = "Very Low"
            advice = "Health status appears excellent. The probability of disease is minimal. Continue your healthy lifestyle."
        elif risk_pct <= 40:
            risk_level = "Stable"
            advice = "Results are currently within the safe range. Annual periodic check-ups are recommended to maintain safety."
        elif risk_pct <= 50:
            risk_level = "Early Warning"
            advice = "Slight indicators require attention. A precautionary review with a specialist is recommended."
        elif risk_pct <= 90:
            risk_level = "High Risk"
            advice = "Warning: Strong indicators of potential disease. Please visit a doctor immediately for lab tests and imaging."
        else:
            risk_level = "Critical"
            advice = "Urgent Alert: High technical probability of disease confirmation. Immediate hospital admission required for diagnostic protocol."

        return {
            'risk_percentage': round(risk_pct, 1),
            'risk_level': risk_level,
            'advice': advice,
            'features_used': self.CANCER_KEYS
        }

    def _predict_fatty_liver(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Fatty liver prediction - purely numerical."""
        X_input = self._extract_features(profile, self.FATTY_KEYS)
        probs = self.models['fatty_liver'].predict_proba(X_input)[0]
        sick_prob = probs[1] * 100  # Probability of class 1 (sick)

        if sick_prob > 50:
            diagnosis = "Sick / High Risk"
            advice = f"Fatty Liver Detected. Probability of Injury: {sick_prob:.1f}%"
        else:
            diagnosis = "Healthy / Low Risk"
            advice = f"Liver Metabolic Health is Stable. Probability of Injury: {sick_prob:.1f}%"

        return {
            'prediction': 1 if sick_prob > 50 else 0,
            'has_fatty_liver': sick_prob > 50,
            'sick_probability': round(sick_prob, 1),
            'confidence': round(sick_prob, 1),
            'diagnosis': diagnosis,
            'advice': advice,
            'features_used': self.FATTY_KEYS
        }

    @staticmethod
    def _platelets_10e9(raw: Any) -> float:
        """
        Platelets in 10^9/L, which is what the HCV models were trained on.

        THIS IS LOAD-BEARING, NOT COSMETIC. The form sends a raw count
        (280000 per microlitre); the Mayo-derived training data holds ~60-720.
        Feeding 280000 straight in puts every one of these models far outside
        its training distribution, and it fails silently -- a prediction still
        comes back, it is simply wrong.

        The previous engine already had this conversion but applied it ONLY to
        the displayed APRI score, while handing the models the raw value
        (verified: index 7 of the old 16-feature vector was 180000.0). Now one
        rule feeds both.
        """
        p = float(raw)
        if p < 10:      # already normalised but scaled down, e.g. 2.8
            return p * 100
        if p > 1000:    # raw count per microlitre, e.g. 280000
            return p / 1000
        return p        # already 10^9/L

    def _hepc_features(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Every value the three HCV models need, computed once.

        Built off `_extract_features` rather than reading `profile` directly, so
        the categorical encoding (gender, ascites, edema's 0/0.5/1 ordinal) stays
        in the one place that already owns it.
        """
        v = self._extract_features(profile, self.HEP_STAGE_KEYS).flatten().tolist()
        bili, chol, alb, cu, alp, ast, tg, plt_raw, pt, age, sex, asc, hep, spi, ede = v

        plt = self._platelets_10e9(plt_raw)

        f = {
            'Bilirubin': bili, 'Cholesterol': chol, 'Albumin': alb, 'Copper': cu,
            'Alk_Phos': alp, 'SGOT': ast, 'Tryglicerides': tg, 'Platelets': plt,
            'Prothrombin': pt, 'Age': age, 'Sex': sex, 'Ascites': asc,
            'Hepatomegaly': hep, 'Spiders': spi, 'Edema': ede,
        }

        # The engineered features, exactly as the model documentation defines
        # them. These are INPUTS to the models here, as well as being reported.
        f['APRI'] = ((ast / 40.0) / plt) * 100 if plt else 0.0
        f['ALBI_Score'] = (math.log10(max(bili, 0.1) * 17.1) * 0.66) + (alb * 10 * -0.085)
        f['Bili_Alb_Ratio'] = (bili / alb) if alb else 0.0
        f['Bilirubin_Albumin'] = bili * alb
        f['Copper_Platelets'] = (cu / plt) if plt else 0.0
        return f

    def _predict_hepatitis(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        HCV three-model flow: complications, then status, then stage.

        The stage model consumes `Status`, so status must run first. See the
        note on HEPC_STAGE_COLS -- this order is forced by the models, not
        chosen.
        """
        f = self._hepc_features(profile)

        # TWO INPUT FORMS, AND THEY ARE NOT INTERCHANGEABLE.
        #
        # The stage and status models are Pipelines whose ColumnTransformer
        # selects columns BY NAME, so they need a DataFrame. The complications
        # model is a bare XGBClassifier whose booster carries feature names as
        # numpy.str_, and under xgboost 3.1.3 passing it a DataFrame raises
        # "data did not contain feature names" -- while a plain array is
        # accepted. Verified both ways; this is not a preference.
        def frame(cols):
            return pd.DataFrame([[f[c] for c in cols]], columns=cols)

        def array(cols):
            return np.array([[f[c] for c in cols]], dtype=float)

        # Step 1: risk of fluid build-up. Independent of everything else.
        complications_pred = float(
            self.models['hepatitis_complications'].predict_proba(array(self.HEPC_COMP_COLS))[0][1]
        )

        # Step 2: mortality risk. No stage among its inputs.
        mortality_risk = float(
            self.models['hepatitis_status'].predict_proba(frame(self.HEPC_STATUS_COLS))[0][1]
        )

        # Step 3: stage, taking the status model's own verdict as its `Status`
        # input. Passing the binary rather than the probability because the
        # training column was binary (C:0 alive / D:1 deceased).
        f['Status'] = 1.0 if mortality_risk > 0.5 else 0.0

        _stage_model = self.models['hepatitis_stage']
        X_stage = frame(self.HEPC_STAGE_COLS)
        _stage_proba = _stage_model.predict_proba(X_stage)[0]
        _stage_classes = [int(c) for c in _stage_model.classes_]
        _stage_class = int(_stage_model.predict(X_stage)[0])

        # Classes are 0/1/2; clinicians count from 1. The displayed stage is the
        # class plus one, and every label downstream follows that -- see
        # _get_stage_description.
        stage_pred = _stage_class + 1
        stage_distribution = {f'Stage {c + 1}': round(float(p) * 100, 1)
                              for c, p in zip(_stage_classes, _stage_proba)}
        stage_confidence_pct = round(
            float(_stage_proba[_stage_classes.index(_stage_class)]) * 100, 1
        )

        apri_score = f['APRI']
        albi_score = f['ALBI_Score']

        # Generate main advice based on mortality risk (Critical if > 0.5)
        if mortality_risk > 0.5:
            advice = "Critical - High survival risk detected. Immediate medical intervention required."
        else:
            advice = "Stable - Liver function is within acceptable range. Continue monitoring."

        return {
            'stage': stage_pred,
            'stage_distribution': stage_distribution,
            'stage_confidence_pct': stage_confidence_pct,
            'complications_risk': round(complications_pred * 100, 1),
            'mortality_risk': round(mortality_risk * 100, 1),
            'stage_description': self._get_stage_description(stage_pred),
            'stage_advice': self._get_stage_advice(stage_pred),
            'complications_advice': self._get_complications_advice(complications_pred),
            'status_advice': self._get_status_advice(mortality_risk),
            'advice': advice,  # Main advice key for UI display
            'apri_score': round(apri_score, 2),
            'albi_score': round(albi_score, 2),
            'risk_level': 'critical' if mortality_risk > 0.8 else 'high' if mortality_risk > 0.5 else 'moderate' if mortality_risk > 0.2 else 'low'
        }

    def _get_stage_description(self, stage: int) -> str:
        """
        Human-readable stage, keyed on the DISPLAYED stage (1-3), not the raw
        model class (0-2).

        Wording is Ali's, given verbatim on 2026-08-10, and is a deliberate
        exception to the plain-English rule in LESSONS.md L-026 -- "Fibrosis"
        and "Cirrhosis" come back here on his explicit instruction. Do not
        "correct" these three strings back to plain English in a later pass.

        The three are a RE-PARTITION of the old five, not the first three of
        them. Stage 3 is cirrhosis, which under the previous mapping would have
        rendered as "F2 - Moderate fibrosis" -- the reason the label change had
        to ship in the same commit as the models.
        """
        descriptions = {
            1: "Stage 1: Early Stage / Mild Fibrosis (F1)",
            2: "Stage 2: Intermediate Stage (F2, F3)",
            3: "Stage 3: Advanced Stage / Liver Cirrhosis (F4)",
        }
        return descriptions.get(stage, f"Stage {stage}")

    def _get_stage_advice(self, stage: int) -> str:
        """Specific advice per displayed stage (1-3)."""
        advice = {
            1: "Early Stage - Mild scarring detected. Often reversible with treatment of the underlying cause and lifestyle change.",
            2: "Intermediate Stage - Scarring is established. Requires medical management to slow progression.",
            3: "Advanced Stage - Cirrhosis. Strict medical intervention and surveillance for complications required.",
        }
        return advice.get(stage, f"Stage {stage} detected.")

    def _get_complications_advice(self, complications_risk: float) -> str:
        """Get advice based on ascites risk percentage."""
        risk_pct = complications_risk * 100
        if risk_pct < 30:
            return "Stable - Fluid balance is currently stable."
        elif risk_pct < 60:
            return "Warning - Early signs of portal hypertension. Monitor abdominal swelling."
        else:
            return "Critical - High risk of fluid accumulation (Ascites). Immediate evaluation required."

    def _get_status_advice(self, mortality_risk: float) -> str:
        """Get advice based on mortality risk percentage."""
        risk_pct = mortality_risk * 100
        if risk_pct <= 20:
            return "Low Risk - Condition stable. Continue preventive protocol."
        elif risk_pct <= 50:
            return "Moderate Risk - Declining liver function. Requires close medical monitoring."
        elif risk_pct <= 80:
            return "High Risk - Advanced liver failure. Radical treatment options should be considered."
        else:
            return "Critical Risk - End-Stage Liver Disease. Immediate intensive care required."

    def _generate_healthy_response(self, gate_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response for healthy patients.

        ADDITIVE, 2026-08-09. This used to build a hardcoded gate dict,
        discarding the real prediction -- so the healthy path carried no
        confidence at all. It also set prediction=0, which by this engine's
        own convention (is_healthy = prediction == 1) means SICK, while
        claiming is_healthy=True in the same dict.

        Passing the real gate_result through fixes both. The parameter
        defaults to None so any existing caller keeps working.
        """
        return {
            'success': True,
            'diagnosis': 'Healthy',
            'confidence': 95,
            'advice': 'Maintain your current healthy lifestyle.',
            'detailed_results': {
                'gate': gate_result if gate_result else {'prediction': 1, 'is_healthy': True}
            },
            'risk_level': 'low'
        }

    def _generate_final_diagnosis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final diagnosis and advice based on all model results."""
        # Determine primary diagnosis based on most severe condition
        diagnoses = []

        # Check cancer risk
        if results['cancer']['risk_percentage'] > 50:
            diagnoses.append(('cancer_high_risk', results['cancer']['risk_percentage']))

        # Check fatty liver
        if results['fatty_liver']['has_fatty_liver']:
            diagnoses.append(('fatty_liver', 80))  # Assume high confidence for deterministic model

        # Check hepatitis
        if results['hepatitis']['mortality_risk'] > 50:
            diagnoses.append(('hepatitis_critical', results['hepatitis']['mortality_risk']))

        if not diagnoses:
            # No major issues detected
            return {
                'diagnosis': 'Normal Liver Function',
                'confidence': 85,
                'advice': 'All liver function tests within normal ranges. Continue routine monitoring.',
                'risk_level': 'low'
            }

        # Sort by severity (highest risk first)
        diagnoses.sort(key=lambda x: x[1], reverse=True)
        primary_diagnosis, confidence = diagnoses[0]

        # Generate advice based on primary diagnosis
        advice = self._generate_advice(primary_diagnosis, results)

        return {
            'diagnosis': self._get_diagnosis_name(primary_diagnosis, results),
            'confidence': int(confidence),
            'advice': advice,
            'risk_level': 'high' if confidence > 70 else 'moderate'
        }

    def _get_diagnosis_name(self, diagnosis_type: str, results: Dict[str, Any]) -> str:
        """Get human-readable diagnosis name."""
        if diagnosis_type == 'cancer_high_risk':
            return f"High Cancer Risk ({results['cancer']['risk_percentage']}%)"
        elif diagnosis_type == 'fatty_liver':
            return "Fatty Liver Disease Detected"
        elif diagnosis_type == 'hepatitis_critical':
            stage = results['hepatitis']['stage']
            return f"Hepatitis Fibrosis (Stage {stage})"
        else:
            return "Liver Condition Detected"

    def _generate_advice(self, diagnosis_type: str, results: Dict[str, Any]) -> str:
        """Generate medical advice based on diagnosis type."""
        if diagnosis_type == 'cancer_high_risk':
            return "High probability detected. Immediate consultation with an oncologist is recommended. Reduce environmental toxins (Smoking/Alcohol)."
        elif diagnosis_type == 'fatty_liver':
            return "Detected signs of NAFLD. Recommended action: Reduce carbohydrate intake and consult a nutritionist."
        elif diagnosis_type == 'hepatitis_critical':
            return "Critical Liver Function. Urgent medical attention required."
        else:
            return "Consult healthcare professional for detailed evaluation."