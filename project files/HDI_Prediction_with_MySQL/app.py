"""Flask web application for predicting Human Development Index (HDI)."""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from flask import Flask, flash, render_template, request, session

from database import DatabaseOperationError, initialize_database, store_prediction_record


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hdi_model.pkl"
FEATURES = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "Gross National Income per Capita",
]
FIELD_LIMITS: Dict[str, Tuple[float, float]] = {
    "Life Expectancy": (20.0, 100.0),
    "Mean Years of Schooling": (0.0, 25.0),
    "Expected Years of Schooling": (0.0, 30.0),
    "Gross National Income per Capita": (0.0, 250000.0),
}

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key-before-deployment"


def load_model():
    """Load the serialized training pipeline once when the application starts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` before starting Flask."
        )
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


model = load_model()
database_available = initialize_database()


def parse_prediction_form(form) -> tuple[str, list[float]]:
    """Validate submitted form values and return country plus numeric features."""
    country = form.get("country", "").strip()
    if not country:
        raise ValueError("Please enter a country or region name.")

    values = []
    for feature in FEATURES:
        field_name = feature.lower().replace(" ", "_")
        try:
            value = float(form.get(field_name, ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be a valid number.") from exc

        lower, upper = FIELD_LIMITS[feature]
        if not lower <= value <= upper:
            raise ValueError(f"{feature} must be between {lower:g} and {upper:g}.")
        values.append(value)
    return country, values


def get_browser_session_token() -> str:
    """Return a stable anonymous browser token for normalized database storage."""
    if "hdi_session_token" not in session:
        session["hdi_session_token"] = str(uuid.uuid4())
    return session["hdi_session_token"]


def persist_prediction(country: str, values: list[float], prediction: float) -> None:
    """Save one request without altering the existing ML response on DB outages."""
    global database_available
    if not database_available:
        return

    try:
        store_prediction_record(
            country_name=country,
            life_expectancy=values[0],
            mean_years_of_schooling=values[1],
            expected_years_of_schooling=values[2],
            gni_per_capita=values[3],
            predicted_hdi=prediction,
            session_token=get_browser_session_token(),
        )
        database_available = True
    except DatabaseOperationError as error:
        database_available = False
        app.logger.warning("Prediction was calculated but not saved to MySQL: %s", error)


@app.route("/")
def home():
    """Render the landing page."""
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Show the prediction form or calculate an HDI score from submitted values."""
    if request.method == "GET":
        return render_template("indexnew.html", limits=FIELD_LIMITS)

    try:
        country, values = parse_prediction_form(request.form)
        feature_frame = pd.DataFrame([values], columns=FEATURES)
        raw_prediction = float(model.predict(feature_frame)[0])
        # HDI is conventionally reported on a 0 to 1 scale.
        prediction = float(np.clip(raw_prediction, 0.0, 1.0))
    except ValueError as error:
        flash(str(error), "error")
        return render_template("indexnew.html", limits=FIELD_LIMITS), 400

    persist_prediction(country, values, prediction)

    return render_template(
        "result.html",
        country=country,
        prediction=prediction,
        inputs=dict(zip(FEATURES, values)),
    )


if __name__ == "__main__":
    app.run(debug=True)
