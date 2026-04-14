from datetime import datetime
from io import BytesIO
import pickle
import textwrap

import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, session, url_for

app = Flask(__name__)
app.secret_key = "heart-disease-dashboard-secret"

model = pickle.load(open("rf_classifier.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
dataset = pd.read_csv("datasets_4123_6408_framingham.csv")

FIELD_LABELS = {
    "male": "Gender",
    "age": "Age",
    "currentSmoker": "Current Smoker",
    "cigsPerDay": "Cigarettes / Day",
    "BPMeds": "BP Medication",
    "prevalentStroke": "Stroke History",
    "prevalentHyp": "Hypertension",
    "diabetes": "Diabetes",
    "totChol": "Total Cholesterol",
    "sysBP": "Systolic BP",
    "diaBP": "Diastolic BP",
    "BMI": "BMI",
    "heartRate": "Heart Rate",
    "glucose": "Glucose",
}

MODEL_COLUMNS = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

SETTINGS_DEFAULTS = {
    "facilityName": "Cardio Insight Center",
    "clinicianName": "Clinical Review Team",
    "riskThreshold": "35",
    "reportNotes": (
        "Use this screening result together with medical history, symptom review, "
        "and clinician evaluation."
    ),
    "themePreference": "system",
    "accentStyle": "teal",
    "showDatasetSummary": "1",
    "showBenchmarkPanel": "1",
    "showRecommendationPanel": "1",
    "showChecklistPanel": "1",
    "compactMode": "0",
}

TEXT_SETTING_KEYS = {
    "facilityName",
    "clinicianName",
    "riskThreshold",
    "reportNotes",
    "themePreference",
    "accentStyle",
}

CHECKBOX_SETTING_KEYS = {
    "showDatasetSummary",
    "showBenchmarkPanel",
    "showRecommendationPanel",
    "showChecklistPanel",
    "compactMode",
}

FORM_DEFAULTS = {
    "patientName": "",
    "male": "1",
    "age": "",
    "currentSmoker": "0",
    "cigsPerDay": "",
    "BPMeds": "0",
    "prevalentStroke": "0",
    "prevalentHyp": "0",
    "diabetes": "0",
    "totChol": "",
    "sysBP": "",
    "diaBP": "",
    "BMI": "",
    "heartRate": "",
    "glucose": "",
}

ACCENT_CHOICES = {"teal", "navy", "rose"}
THEME_CHOICES = {"system", "light", "dark"}


def now_display():
    return datetime.now().strftime("%d %b %Y, %I:%M %p")


def now_report_display():
    return datetime.now().strftime("%d %B %Y, %I:%M %p")


def normalize_gender(value):
    value = str(value).strip().lower()
    return 1 if value in {"1", "male", "m", "true"} else 0


def normalize_bool(value):
    value = str(value).strip().lower()
    return 1 if value in {"1", "yes", "y", "true", "on"} else 0


def get_text(form, key, label):
    value = form.get(key, "").strip()
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def get_float(form, key):
    value = form.get(key, "").strip()
    if value == "":
        raise ValueError(f"{FIELD_LABELS.get(key, key)} is required.")
    return float(value)


def get_int(form, key):
    value = form.get(key, "").strip()
    if value == "":
        raise ValueError(f"{FIELD_LABELS.get(key, key)} is required.")
    return int(float(value))


def clamp_float(value, minimum, maximum):
    return max(minimum, min(maximum, float(value)))


def get_saved_settings():
    saved = session.get("dashboard_settings", {})
    settings = SETTINGS_DEFAULTS.copy()
    for key, default_value in SETTINGS_DEFAULTS.items():
        candidate = saved.get(key, default_value)
        if key in CHECKBOX_SETTING_KEYS:
            settings[key] = "1" if str(candidate) == "1" else "0"
        else:
            text = str(candidate).strip()
            settings[key] = text or default_value

    if settings["themePreference"] not in THEME_CHOICES:
        settings["themePreference"] = SETTINGS_DEFAULTS["themePreference"]
    if settings["accentStyle"] not in ACCENT_CHOICES:
        settings["accentStyle"] = SETTINGS_DEFAULTS["accentStyle"]
    settings["riskThreshold"] = str(
        int(round(clamp_float(settings["riskThreshold"], 5, 95)))
    )
    return settings


def parse_settings(form):
    settings = get_saved_settings()
    for key in TEXT_SETTING_KEYS:
        value = form.get(key, settings[key]).strip()
        if value:
            settings[key] = value

    settings["riskThreshold"] = str(
        int(round(clamp_float(settings["riskThreshold"], 5, 95)))
    )

    if settings["themePreference"] not in THEME_CHOICES:
        settings["themePreference"] = SETTINGS_DEFAULTS["themePreference"]
    if settings["accentStyle"] not in ACCENT_CHOICES:
        settings["accentStyle"] = SETTINGS_DEFAULTS["accentStyle"]

    for key in CHECKBOX_SETTING_KEYS:
        settings[key] = "1" if form.get(key) else "0"
    return settings


def capture_form_state(form=None):
    state = FORM_DEFAULTS.copy()
    if form is None:
        return state
    for key in state:
        state[key] = str(form.get(key, state[key])).strip()
    return state


def dataset_overview():
    return {
        "records": int(len(dataset)),
        "positive_rate": round(float(dataset["TenYearCHD"].mean() * 100), 1),
        "avg_age": round(float(dataset["age"].mean()), 1),
        "avg_sysbp": round(float(dataset["sysBP"].mean()), 1),
        "avg_cholesterol": round(float(dataset["totChol"].mean()), 1),
        "smoker_rate": round(float(dataset["currentSmoker"].mean() * 100), 1),
        "hypertension_rate": round(float(dataset["prevalentHyp"].mean() * 100), 1),
        "diabetes_rate": round(float(dataset["diabetes"].mean() * 100), 1),
    }


def build_feature_array(form_values):
    return pd.DataFrame(
        [[
            normalize_gender(form_values["male"]),
            form_values["age"],
            normalize_bool(form_values["currentSmoker"]),
            form_values["cigsPerDay"],
            normalize_bool(form_values["BPMeds"]),
            normalize_bool(form_values["prevalentStroke"]),
            normalize_bool(form_values["prevalentHyp"]),
            normalize_bool(form_values["diabetes"]),
            form_values["totChol"],
            form_values["sysBP"],
            form_values["diaBP"],
            form_values["BMI"],
            form_values["heartRate"],
            form_values["glucose"],
        ]],
        columns=MODEL_COLUMNS,
    )


def describe_bp(sys_bp, dia_bp):
    if sys_bp >= 140 or dia_bp >= 90:
        return "High"
    if sys_bp >= 120 or dia_bp >= 80:
        return "Elevated"
    return "Stable"


def describe_bmi(bmi):
    if bmi >= 30:
        return "Obesity Range"
    if bmi >= 25:
        return "Overweight Range"
    if bmi >= 18.5:
        return "Healthy Range"
    return "Low Range"


def describe_glucose(glucose):
    if glucose >= 126:
        return "High"
    if glucose >= 100:
        return "Borderline"
    return "Normal"


def describe_cholesterol(cholesterol):
    if cholesterol >= 240:
        return "High"
    if cholesterol >= 200:
        return "Borderline"
    return "Desirable"


def determine_priority(probability, threshold, bp_status, glucose_status):
    if probability >= 65 or bp_status == "High" or glucose_status == "High":
        return "Immediate Review"
    if probability >= threshold:
        return "Priority Follow-up"
    return "Routine Monitoring"


def build_observations(form_values, result):
    observations = []

    if form_values["sysBP"] >= 140 or form_values["diaBP"] >= 90:
        observations.append(
            {
                "title": "Blood pressure needs attention",
                "detail": (
                    f"Recorded pressure is {int(form_values['sysBP'])}/{int(form_values['diaBP'])} mmHg, "
                    "which is above the preferred screening range."
                ),
                "level": "critical",
            }
        )

    if form_values["totChol"] >= 240:
        observations.append(
            {
                "title": "Cholesterol is in the high range",
                "detail": (
                    f"Total cholesterol is {int(form_values['totChol'])} mg/dL. A clinician-led lipid review "
                    "would be appropriate."
                ),
                "level": "alert",
            }
        )

    if normalize_bool(form_values["currentSmoker"]):
        observations.append(
            {
                "title": "Smoking status increases cardiac load",
                "detail": (
                    f"Current smoking is marked as active with {int(form_values['cigsPerDay'])} cigarettes per day."
                ),
                "level": "alert",
            }
        )

    if form_values["glucose"] >= 126 or normalize_bool(form_values["diabetes"]):
        observations.append(
            {
                "title": "Glucose monitoring should be prioritised",
                "detail": (
                    f"Glucose is {int(form_values['glucose'])} mg/dL and diabetes history must be reviewed carefully."
                ),
                "level": "alert",
            }
        )

    if form_values["BMI"] >= 30:
        observations.append(
            {
                "title": "Weight profile may be influencing long-term risk",
                "detail": (
                    f"BMI is {form_values['BMI']:.1f}, which sits in the obesity range for screening purposes."
                ),
                "level": "watch",
            }
        )

    if form_values["age"] >= 55:
        observations.append(
            {
                "title": "Age band warrants closer preventive screening",
                "detail": (
                    f"Age {form_values['age']} falls into a higher-risk cohort for ten-year cardiovascular events."
                ),
                "level": "watch",
            }
        )

    if not observations:
        observations.append(
            {
                "title": "Submitted vitals are relatively stable",
                "detail": (
                    "No major screening flag stands out from the submitted values, but preventive follow-up "
                    "should still continue."
                ),
                "level": "stable",
            }
        )

    if result["probability"] >= result["threshold"]:
        observations.insert(
            0,
            {
                "title": "Model probability crossed the dashboard alert threshold",
                "detail": (
                    f"Estimated ten-year risk is {result['probability']:.2f}%, above the configured threshold of "
                    f"{result['threshold']:.0f}%."
                ),
                "level": "critical",
            },
        )

    return observations[:5]


def build_recommendations(form_values, result):
    recommendations = []

    if result["priority_level"] == "Immediate Review":
        recommendations.append(
            "Arrange clinician review soon and pair the model result with symptoms, history, ECG, and medication review."
        )
    else:
        recommendations.append(
            "Continue scheduled preventive screening and compare this result against historical measurements where available."
        )

    if form_values["sysBP"] >= 120 or form_values["diaBP"] >= 80:
        recommendations.append(
            "Track blood pressure readings across several days to confirm whether the elevation is persistent."
        )

    if form_values["totChol"] >= 200:
        recommendations.append(
            "Consider lipid profile follow-up and dietary counselling to address cholesterol exposure."
        )

    if normalize_bool(form_values["currentSmoker"]):
        recommendations.append(
            "Support smoking cessation planning because tobacco exposure materially raises cardiovascular risk."
        )

    if form_values["glucose"] >= 100 or normalize_bool(form_values["diabetes"]):
        recommendations.append(
            "Review glucose control, HbA1c, and diabetes management targets with a clinician."
        )

    if form_values["BMI"] >= 25:
        recommendations.append(
            "Add a structured activity and nutrition plan that focuses on gradual, sustainable risk reduction."
        )

    return recommendations[:5]


def build_checklist(form_values, result):
    return [
        {
            "label": "Patient identity and screening inputs confirmed",
            "status": "complete",
        },
        {
            "label": "Blood pressure follow-up scheduled",
            "status": "action" if result["bp_status"] != "Stable" else "monitor",
        },
        {
            "label": "Lipid management discussed",
            "status": "action" if result["cholesterol_status"] != "Desirable" else "monitor",
        },
        {
            "label": "Glucose review planned",
            "status": "action" if result["glucose_status"] != "Normal" else "monitor",
        },
        {
            "label": "Lifestyle support plan documented",
            "status": "action"
            if normalize_bool(form_values["currentSmoker"]) or form_values["BMI"] >= 25
            else "monitor",
        },
    ]


def build_benchmarks(form_values):
    benchmark_fields = [
        ("age", "years"),
        ("sysBP", "mmHg"),
        ("totChol", "mg/dL"),
        ("BMI", "kg/m2"),
        ("glucose", "mg/dL"),
    ]
    cards = []

    for key, unit in benchmark_fields:
        series = dataset[key].dropna()
        patient_value = float(form_values[key])
        average = float(series.mean())
        percentile = int(round(float((series <= patient_value).mean() * 100)))
        difference = round(patient_value - average, 1)

        if difference > 0:
            direction = "above"
        elif difference < 0:
            direction = "below"
        else:
            direction = "at"

        cards.append(
            {
                "label": FIELD_LABELS[key],
                "patient_value": round(patient_value, 1),
                "average": round(average, 1),
                "difference": abs(difference),
                "direction": direction,
                "percentile": percentile,
                "unit": unit,
            }
        )

    return cards


def compute_health_score(form_values, result):
    score = 100

    if result["probability"] >= result["threshold"]:
        score -= 20
    if result["bp_status"] == "High":
        score -= 15
    elif result["bp_status"] == "Elevated":
        score -= 8
    if result["cholesterol_status"] == "High":
        score -= 10
    elif result["cholesterol_status"] == "Borderline":
        score -= 5
    if result["glucose_status"] == "High":
        score -= 10
    elif result["glucose_status"] == "Borderline":
        score -= 5
    if normalize_bool(form_values["currentSmoker"]):
        score -= 10
    if form_values["BMI"] >= 30:
        score -= 10
    elif form_values["BMI"] >= 25:
        score -= 5

    return max(20, score)


def safe_filename_part(value):
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower())
    return cleaned.strip("-") or "patient"


def pdf_escape(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_pdf_text(text, width=88):
    normalized = " ".join(str(text).split())
    return textwrap.wrap(normalized, width=width) or [""]


def build_report_sections(settings, patient_values, result, insights):
    return [
        (
            "Patient Details",
            [
                f"Patient Name: {patient_values['patientName']}",
                f"Gender: {patient_values['male_label']}",
                f"Age: {patient_values['age']}",
                f"Generated At: {result['report_generated_at']}",
            ],
        ),
        (
            "Risk Summary",
            [
                f"Estimated 10-year risk: {result['probability']:.2f}%",
                f"Risk status: {result['risk_status']}",
                f"Priority level: {result['priority_level']}",
                f"Health score: {result['health_score']}/100",
                f"Alert threshold: {result['threshold']:.0f}%",
                result["summary"],
            ],
        ),
        (
            "Vital Measurements",
            [f"{item['label']}: {item['value']} ({item['status']})" for item in insights["vitals"]],
        ),
        (
            "Clinical Observations",
            [f"{item['title']}: {item['detail']}" for item in insights["observations"]],
        ),
        (
            "Recommended Actions",
            [f"- {item}" for item in insights["recommendations"]],
        ),
        (
            "Clinical Notes",
            [
                settings["reportNotes"],
                "This report supports screening and should be reviewed alongside clinician judgment and additional diagnostic evaluation.",
            ],
        ),
    ]


def build_pdf_report(settings, patient_values, result, insights):
    page_width = 612
    page_height = 792
    margin_x = 54
    margin_top = 64
    margin_bottom = 54
    line_height = 16
    pages = []
    current_commands = []
    y = page_height - margin_top

    def start_page():
        return []

    def flush_page():
        nonlocal current_commands, y
        pages.append("\n".join(current_commands))
        current_commands = start_page()
        y = page_height - margin_top

    def ensure_space(lines_needed=1):
        nonlocal y
        needed_height = lines_needed * line_height
        if y - needed_height < margin_bottom:
            flush_page()

    def add_text(text, x, y_pos, font="F1", size=11):
        current_commands.append(f"BT /{font} {size} Tf 1 0 0 1 {x} {y_pos} Tm ({pdf_escape(text)}) Tj ET")

    def add_rule(y_pos):
        current_commands.append(f"{margin_x} {y_pos} m {page_width - margin_x} {y_pos} l S")

    current_commands = start_page()

    add_text(settings["facilityName"], margin_x, y, font="F2", size=20)
    y -= 24
    add_text("Heart Disease Detection Report", margin_x, y, font="F2", size=16)
    y -= 18
    add_text(f"Patient: {patient_values['patientName']}", margin_x, y, font="F1", size=12)
    y -= 16
    add_text(f"Prepared by: {settings['clinicianName']}  |  Date: {result['report_generated_at']}", margin_x, y, font="F1", size=10)
    y -= 14
    add_rule(y)
    y -= 24

    summary_lines = [
        f"Estimated 10-year risk: {result['probability']:.2f}%",
        f"Risk status: {result['risk_status']}",
        f"Priority: {result['priority_level']}",
        f"Follow-up: {result['follow_up_window']}",
    ]
    ensure_space(len(summary_lines) + 3)
    add_text("Executive Summary", margin_x, y, font="F2", size=13)
    y -= 20
    for line in summary_lines:
        add_text(line, margin_x, y, font="F1", size=11)
        y -= line_height
    y -= 8

    for section_title, items in build_report_sections(settings, patient_values, result, insights):
        wrapped_lines = []
        for item in items:
            wrapped_lines.extend(wrap_pdf_text(item))
            wrapped_lines.append("")
        if wrapped_lines and wrapped_lines[-1] == "":
            wrapped_lines.pop()

        ensure_space(len(wrapped_lines) + 3)
        add_text(section_title, margin_x, y, font="F2", size=13)
        y -= 18
        for line in wrapped_lines:
            if line == "":
                y -= 6
                continue
            add_text(line, margin_x, y, font="F1", size=10)
            y -= line_height
        y -= 10

    add_rule(margin_bottom - 8)
    add_text("Generated by Heart Predictor Pro", margin_x, margin_bottom - 24, font="F1", size=9)
    pages.append("\n".join(current_commands))

    objects = []

    def add_object(body):
        objects.append(body)
        return len(objects)

    font_regular_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    page_ids = []

    for content in pages:
        stream = f"<< /Length {len(content.encode('latin-1', 'replace'))} >>\nstream\n{content}\nendstream"
        content_id = add_object(stream)
        page_body = (
            "<< /Type /Page /Parent {parent} 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> >>"
        )
        page_ids.append(add_object(page_body))

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>")

    for page_id in page_ids:
        objects[page_id - 1] = objects[page_id - 1].replace("{parent}", str(pages_id))

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n{body}\nendobj\n".encode("latin-1", "replace"))

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode(
            "latin-1"
        )
    )
    return bytes(pdf)


def build_result(form_values, settings):
    scaled_features = scaler.transform(build_feature_array(form_values))
    prediction = int(model.predict(scaled_features)[0])
    probability = float(model.predict_proba(scaled_features)[0][1] * 100)
    threshold = float(settings["riskThreshold"])
    bp_status = describe_bp(form_values["sysBP"], form_values["diaBP"])
    bmi_status = describe_bmi(form_values["BMI"])
    glucose_status = describe_glucose(form_values["glucose"])
    cholesterol_status = describe_cholesterol(form_values["totChol"])
    probability_gap = round(abs(probability - threshold), 2)

    if probability >= threshold:
        risk_status = "High Risk"
        summary = (
            "The submitted profile crossed the alert threshold, so the case should be reviewed as a higher-priority screening result."
        )
    else:
        risk_status = "Lower Risk"
        summary = (
            "The submitted profile is below the current alert threshold, though preventive monitoring should still continue."
        )

    if prediction == 1 and probability < threshold:
        summary = (
            "The classifier signalled elevated risk, even though the configured dashboard threshold is higher than the model probability."
        )

    priority_level = determine_priority(probability, threshold, bp_status, glucose_status)
    health_score = compute_health_score(
        form_values,
        {
            "probability": probability,
            "threshold": threshold,
            "bp_status": bp_status,
            "cholesterol_status": cholesterol_status,
            "glucose_status": glucose_status,
        },
    )

    return {
        "prediction_flag": prediction,
        "probability": round(probability, 2),
        "threshold": round(threshold, 2),
        "probability_gap": probability_gap,
        "risk_status": risk_status,
        "summary": summary,
        "priority_level": priority_level,
        "bp_status": bp_status,
        "bmi_status": bmi_status,
        "glucose_status": glucose_status,
        "cholesterol_status": cholesterol_status,
        "smoking_status": "Active Smoker" if normalize_bool(form_values["currentSmoker"]) else "Non-Smoker",
        "generated_at": now_display(),
        "report_generated_at": now_report_display(),
        "health_score": health_score,
        "care_window": "Clinical review within 7 days" if probability >= threshold else "Continue routine preventive follow-up",
        "follow_up_window": "Re-screen in 1-3 months" if probability >= threshold else "Re-screen in 6-12 months",
        "status_class": "high" if probability >= threshold else "low",
    }


def parse_form_values(form):
    return {
        "patientName": get_text(form, "patientName", "Patient name"),
        "male": form.get("male", "1"),
        "age": get_int(form, "age"),
        "currentSmoker": form.get("currentSmoker", "0"),
        "cigsPerDay": get_float(form, "cigsPerDay"),
        "BPMeds": form.get("BPMeds", "0"),
        "prevalentStroke": form.get("prevalentStroke", "0"),
        "prevalentHyp": form.get("prevalentHyp", "0"),
        "diabetes": form.get("diabetes", "0"),
        "totChol": get_float(form, "totChol"),
        "sysBP": get_float(form, "sysBP"),
        "diaBP": get_float(form, "diaBP"),
        "BMI": get_float(form, "BMI"),
        "heartRate": get_float(form, "heartRate"),
        "glucose": get_float(form, "glucose"),
    }


def serialize_form_values(form_values):
    serialized = capture_form_state()
    for key, value in form_values.items():
        serialized[key] = value

    serialized["male_label"] = "Male" if normalize_gender(form_values["male"]) else "Female"
    for key in ["currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]:
        serialized[f"{key}_label"] = "Yes" if normalize_bool(form_values[key]) else "No"
    return serialized


def build_dashboard_context(form_values, result):
    if not form_values or not result:
        return None

    observations = build_observations(form_values, result)
    recommendations = build_recommendations(form_values, result)
    checklist = build_checklist(form_values, result)
    benchmarks = build_benchmarks(form_values)

    return {
        "observations": observations,
        "recommendations": recommendations,
        "checklist": checklist,
        "benchmarks": benchmarks,
        "vitals": [
            {"label": "Blood Pressure", "value": f"{int(form_values['sysBP'])}/{int(form_values['diaBP'])} mmHg", "status": result["bp_status"]},
            {"label": "BMI", "value": f"{form_values['BMI']:.1f}", "status": result["bmi_status"]},
            {"label": "Glucose", "value": f"{int(form_values['glucose'])} mg/dL", "status": result["glucose_status"]},
            {"label": "Cholesterol", "value": f"{int(form_values['totChol'])} mg/dL", "status": result["cholesterol_status"]},
            {"label": "Smoking", "value": result["smoking_status"], "status": f"{int(form_values['cigsPerDay'])} cigarettes/day"},
        ],
    }


def render_dashboard_page(settings, form_values=None, result=None, error=None):
    return render_template(
        "index.html",
        dashboard=dataset_overview(),
        settings=settings,
        form_values=form_values or capture_form_state(),
        result=result,
        error=error,
        field_labels=FIELD_LABELS,
        insights=build_dashboard_context(form_values, result),
        settings_status=request.args.get("settings_status", ""),
        generated_now=now_display(),
    )


@app.route("/")
def index():
    settings = get_saved_settings()
    return render_dashboard_page(settings)


@app.route("/settings", methods=["POST"])
def settings_route():
    settings = parse_settings(request.form)
    session["dashboard_settings"] = settings
    return redirect(url_for("index", settings_status="saved"))


@app.route("/settings/reset", methods=["POST"])
def settings_reset_route():
    session["dashboard_settings"] = SETTINGS_DEFAULTS.copy()
    return redirect(url_for("index", settings_status="reset"))


@app.route("/predict", methods=["POST"])
def predict_route():
    settings = parse_settings(request.form)
    session["dashboard_settings"] = settings

    try:
        parsed_form_values = parse_form_values(request.form)
        result = build_result(parsed_form_values, settings)
        form_payload = serialize_form_values(parsed_form_values)
        error = None
    except ValueError as exc:
        result = None
        form_payload = capture_form_state(request.form)
        error = str(exc)

    return render_dashboard_page(
        settings=settings,
        form_values=form_payload,
        result=result,
        error=error,
    )


@app.route("/report", methods=["POST"])
def report_route():
    settings = parse_settings(request.form)
    session["dashboard_settings"] = settings

    try:
        form_values = parse_form_values(request.form)
    except ValueError as exc:
        return render_dashboard_page(
            settings=settings,
            form_values=capture_form_state(request.form),
            result=None,
            error=str(exc),
        )

    result = build_result(form_values, settings)
    patient_values = serialize_form_values(form_values)
    insights = build_dashboard_context(form_values, result)
    report_pdf = build_pdf_report(settings, patient_values, result, insights)
    report_stream = BytesIO(report_pdf)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"heart-risk-report-{safe_filename_part(form_values['patientName'])}-{timestamp}.pdf"
    return send_file(
        report_stream,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
