"""Prediksi risiko keterlambatan pembayaran invoice (contoh data sintetis).

Sistem mengklasifikasikan invoice per transaksi. Target ``Is_Late`` bernilai 1
jika invoice belum dibayar sampai 30 hari setelah tanggal jatuh tempo.
Fitur yang dipakai hanya informasi yang tersedia saat invoice diterbitkan.

Jalankan dengan: python late_payment_risk.py
Dependensi: pandas, numpy, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = [
    "InvoiceAmount",
    "CreditTerms_Days",
    "PriorLatePayments",
]
CATEGORICAL_FEATURES = ["PaymentMethod"]


def create_synthetic_invoices(
    n_samples: int = 1800, random_state: int = 42
) -> pd.DataFrame:
    """Buat data invoice sintetis dengan pola risiko yang dapat dipelajari."""
    rng = np.random.default_rng(random_state)
    invoice_dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")
    amounts = rng.integers(1_000, 50_001, size=n_samples)
    terms = rng.choice([15, 30, 45, 60], size=n_samples)
    methods = rng.choice(
        ["Bank Transfer", "Credit Card", "Cash"],
        size=n_samples,
        p=[0.55, 0.30, 0.15],
    )
    prior_lates = rng.integers(0, 5, size=n_samples)

    # Label sintetis dipengaruhi faktor risiko agar proses belajar model bermakna.
    score = (
        -3.7
        + 0.65 * prior_lates
        + 0.35 * (amounts >= 35_000)
        + 0.25 * (terms == 60)
        + 0.20 * (methods == "Bank Transfer")
    )
    late_probability = 1 / (1 + np.exp(-score))
    is_late = rng.binomial(1, late_probability)

    return pd.DataFrame(
        {
            "InvoiceDate": invoice_dates,
            "InvoiceAmount": amounts,
            "CreditTerms_Days": terms,
            "PaymentMethod": methods,
            "PriorLatePayments": prior_lates,
            "Is_Late": is_late,
        }
    )


def build_preprocessor() -> ColumnTransformer:
    """Skalakan fitur numerik dan ubah kategori menjadi kolom indikator."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate_models(df: pd.DataFrame) -> pd.DataFrame:
    """Latih dan bandingkan model dengan pemisahan data berdasarkan waktu."""
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    split_index = int(len(df) * 0.8)

    # Data masa depan hanya dipakai untuk pengujian, bukan untuk pelatihan.
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, y_train = train[features], train["Is_Late"]
    x_test, y_test = test[features], test["Is_Late"]

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=42,
        ),
    }

    results = []
    for name, classifier in models.items():
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["F1", "Recall"], ascending=False
    )
    late_rate = y_test.mean() * 100

    print("--- PREDIKSI RISIKO KETERLAMBATAN PEMBAYARAN ---")
    print(f"Jumlah invoice: {len(df)}")
    print(f"Data latih / uji: {len(train)} / {len(test)} (berdasarkan waktu)")
    print(f"Late rate pada data uji: {late_rate:.2f}%")
    print("\nPerbandingan model (diurutkan berdasarkan F1, lalu Recall):")
    print(
        results_df.to_string(
            index=False,
            formatters={
                metric: "{:.3f}".format
                for metric in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC-AUC",
                ]
            },
        )
    )
    print(
        "\nModel dengan F1 tertinggi pada data uji: "
        f"{results_df.iloc[0]['Model']}"
    )
    print(
        "Catatan: data dan hasil ini sintetis. Validasi dengan data historis "
        "sebelum dipakai untuk keputusan bisnis."
    )

    return results_df


def main() -> None:
    invoices = create_synthetic_invoices()
    evaluate_models(invoices)


if __name__ == "__main__":
    main()
