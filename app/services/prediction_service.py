from app.services.feature_builder import build_features
from app.services.model_loader import (
    get_model,
    VALID_MODEL_TYPES
)
from app.services.shap_service import get_top_factors

from app.rag.retriever import retrieve_context
from app.rag.prompts import build_explanation_prompt
from app.rag.llm import generate_response


def predict_patient(data):
    """
    Main prediction pipeline.

    Flow:
    Patient Data
        ↓
    Feature Engineering
        ↓
    Model Prediction
        ↓
    SHAP Explanation
        ↓
    RAG Retrieval
        ↓
    Prompt Construction
        ↓
    Groq LLM Explanation
        ↓
    API Response
    """

    # ---------------- VALIDATE MODEL ----------------

    if data.model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            "Invalid model_type"
        )

    # ---------------- FEATURE BUILDING ----------------
    # Converts incoming patient data into
    # the exact feature vector expected by the model.

    features, feature_names = build_features(data)

    # ---------------- MODEL SELECTION ----------------

    model = get_model(data.model_type)

    # ---------------- PREDICTION ----------------

    prediction = model.predict(features)[0]

    # Probability of Insulin Resistance (Class 1)

    probability = model.predict_proba(features)[0][1]

    # ---------------- LABEL ----------------

    label = (
        "Insulin Resistant"
        if prediction == 1
        else "Normal"
    )

    # ---------------- RISK CATEGORY ----------------

    risk_category = (
        "Low Risk"
        if probability < 0.31
        else "Moderate Risk"
        if probability < 0.61
        else "High Risk"
    )

    # ---------------- SHAP EXPLANATION ----------------

    top_factors, contributions = get_top_factors(
        data.model_type,
        features,
        feature_names
    )

    # ---------------- RAG QUERY ----------------
    # Use the most influential features
    # to retrieve relevant medical context.

    keywords = " ".join(
        feature
        for feature, _ in contributions[:3]
    )

    query = (
        f"{label} "
        f"insulin resistance "
        f"{keywords}"
    )

    # ---------------- RETRIEVE CONTEXT ----------------

    context = retrieve_context(
        query
    )

    # ---------------- BUILD PROMPT ----------------

    prompt = build_explanation_prompt(
        label=label,
        risk_category=risk_category,
        risk_probability=round(
            float(probability) * 100,
            2
        ),
        contributions=contributions,
        context=context
    )

    # ---------------- GENERATE LLM RESPONSE ----------------

    explanation = generate_response(
        prompt
    )

    # ---------------- FINAL RESPONSE ----------------

    return {
        "prediction": int(prediction),
        "label": label,
        "risk_category": risk_category,
        "risk_probability": round(
            float(probability) * 100,
            2
        ),
        "top_risk_factors": top_factors,
        "explanation": explanation
    }