
from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "models" / "model_comparison.joblib"

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
)


# ==================== EXPLANATION RULES ====================

def explain_message(text):
    """Detect suspicious patterns using transparent text rules."""
    text = str(text)
    message = text.lower()
    signals = []

    rules = {
        "Suspicious link": [
            r"https?://\S+",
            r"\bwww\.\S+",
        ],
        "Urgent language": [
            r"\burgent\b",
            r"\bimmediately\b",
            r"\bact now\b",
            r"\blimited time\b",
            r"\bverify now\b",
            r"\bexpires?\b",
        ],
        "Prize or reward claim": [
            r"\bwinner\b",
            r"\bwon\b",
            r"\bprize\b",
            r"\bfree gift\b",
            r"\bclaim your\b",
            r"\blottery\b",
        ],
        "Sensitive information": [
            r"\bpassword\b",
            r"\bbank details\b",
            r"\bcredit card\b",
            r"\botp\b",
            r"\bone time password\b",
            r"\bpin number\b",
        ],
        "Payment request": [
            r"\btransfer money\b",
            r"\bsend money\b",
            r"\bpay now\b",
            r"\bprocessing fee\b",
            r"\bclaim fee\b",
        ],
    }

    for category, patterns in rules.items():
        if any(re.search(pattern, message) for pattern in patterns):
            signals.append(category)

    letters = [char for char in text if char.isalpha()]

    if len(letters) >= 12:
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters)
        )
        if uppercase_ratio >= 0.65:
            signals.append("Excessive capitalization")

    return signals


# ==================== LOAD TRAINED MODELS ====================

@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


st.title("🛡️ SpamShield AI")
st.caption(
    "AI-powered message classification and dataset analysis"
)

if not BUNDLE_PATH.exists():
    st.error(
        "Trained model results are missing. Run "
        "`python train_model.py` in the project terminal first."
    )
    st.stop()

try:
    bundle = load_bundle()
    models = bundle["models"]
    results = bundle["results"]
    best_name = bundle["best_model"]
except Exception as error:
    st.error(f"Could not load the trained models: {error}")
    st.stop()

st.info(
    "This model was trained on SMS messages. "
    "It has not been validated on a dedicated email dataset."
)

tab_dataset, tab_single, tab_evaluation = st.tabs(
    [
        "📂 Dataset Analyzer",
        "📩 Single Message",
        "📊 Model Evaluation",
    ]
)


# ==================== DATASET ANALYZER ====================

with tab_dataset:
    st.subheader("Analyze an entire message dataset")

    st.write(
        "Upload a CSV file containing messages. The application "
        "will predict each message, estimate its spam probability, "
        "identify suspicious patterns, and let you export the results."
    )

    with st.expander("View CSV format example"):
        st.code(
            'message\n'
            '"URGENT! Claim your prize at www.example.com"\n'
            '"Hi, I will meet you after class."\n'
            '"Your account needs verification."',
            language="text",
        )

    uploaded_file = st.file_uploader(
        "Upload your message dataset",
        type=["csv"],
        key="dataset_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8-sig",
                on_bad_lines="error",
            )

            if df.empty:
                st.warning("The uploaded CSV contains no data rows.")

            elif len(df.columns) == 0:
                st.error("No columns were found in the CSV.")

            else:
                column_names = list(df.columns)

                preferred_names = {
                    "message",
                    "text",
                    "body",
                    "content",
                    "sms",
                    "email",
                    "message_text",
                    "email_text",
                }

                matching_columns = [
                    col for col in column_names
                    if str(col).strip().lower() in preferred_names
                ]

                default_index = (
                    column_names.index(matching_columns[0])
                    if matching_columns
                    else 0
                )

                message_column = st.selectbox(
                    "Select the column containing message text",
                    column_names,
                    index=default_index,
                    key="dataset_message_column",
                )

                st.write(f"**Dataset rows:** {len(df):,}")
                st.write(f"**Columns:** {len(df.columns)}")

                if st.button(
                    "🔍 Analyze Dataset",
                    type="primary",
                    key="run_dataset_analysis",
                ):
                    messages = df[message_column].fillna("").astype(str)
                    valid_mask = messages.str.strip().ne("")
                    valid_messages = messages[valid_mask].tolist()

                    if not valid_messages:
                        st.error(
                            "The selected column contains no "
                            "non-empty messages."
                        )
                    else:
                        model = models[best_name]

                        with st.spinner(
                            f"Analyzing {len(valid_messages):,} messages..."
                        ):
                            predictions = model.predict(valid_messages)
                            probabilities = model.predict_proba(
                                valid_messages
                            )

                        classes = list(model.classes_)
                        spam_index = classes.index("spam")
                        spam_scores = probabilities[:, spam_index]

                        output = df.copy()
                        output["Prediction"] = "Not analyzed - empty"
                        output["Spam Probability (%)"] = np.nan
                        output["Suspicious Signals"] = ""

                        row_positions = np.flatnonzero(
                            valid_mask.to_numpy()
                        )

                        output.iloc[
                            row_positions,
                            output.columns.get_loc("Prediction"),
                        ] = predictions

                        output.iloc[
                            row_positions,
                            output.columns.get_loc(
                                "Spam Probability (%)"
                            ),
                        ] = np.round(spam_scores * 100, 2)

                        signal_results = [
                            "; ".join(explain_message(message))
                            or "No rule-based signals detected"
                            for message in valid_messages
                        ]

                        output.iloc[
                            row_positions,
                            output.columns.get_loc("Suspicious Signals"),
                        ] = signal_results

                        st.session_state["dataset_results"] = output
                        st.session_state["dataset_model"] = best_name
                        st.session_state["dataset_source_name"] = (
                            uploaded_file.name
                        )

        except (
            UnicodeDecodeError,
            pd.errors.ParserError,
            ValueError,
        ) as error:
            st.error(
                "The CSV could not be read. Check that it is a valid "
                f"CSV with a header row. Details: {error}"
            )
        except Exception as error:
            st.error(f"Dataset analysis failed: {error}")

    # Show results after analysis, including after Streamlit reruns.
    if "dataset_results" in st.session_state:
        output = st.session_state["dataset_results"]

        # Keep results tied to the currently uploaded file.
        if (
            uploaded_file is not None
            and st.session_state.get("dataset_source_name")
            == uploaded_file.name
        ):
            valid_output = output[
                output["Prediction"].isin(["spam", "ham"])
            ]

            spam_count = int(
                (valid_output["Prediction"] == "spam").sum()
            )
            ham_count = int(
                (valid_output["Prediction"] == "ham").sum()
            )
            total = len(valid_output)
            empty_count = len(output) - total

            st.divider()
            st.subheader("Dataset analysis summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Analyzed", f"{total:,}")
            c2.metric("Likely Spam", f"{spam_count:,}")
            c3.metric("Likely Legitimate", f"{ham_count:,}")
            c4.metric(
                "Spam Share",
                f"{spam_count / total:.1%}" if total else "0.0%",
            )

            if empty_count:
                st.caption(
                    f"{empty_count:,} rows with empty messages "
                    "were not analyzed."
                )

            st.subheader("Filter and inspect messages")

            filter_choice = st.selectbox(
                "Show messages",
                [
                    "All messages",
                    "Spam only",
                    "Legitimate only",
                    "Empty / not analyzed",
                ],
                key="dataset_filter",
            )

            if filter_choice == "Spam only":
                filtered = output[output["Prediction"] == "spam"]
            elif filter_choice == "Legitimate only":
                filtered = output[output["Prediction"] == "ham"]
            elif filter_choice == "Empty / not analyzed":
                filtered = output[
                    output["Prediction"] == "Not analyzed - empty"
                ]
            else:
                filtered = output

            st.write(f"**Rows displayed:** {len(filtered):,}")

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Download your classified dataset")

            csv_data = output.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "⬇️ Download All Results as CSV",
                data=csv_data,
                file_name="spamshield_classified_dataset.csv",
                mime="text/csv",
                key="download_dataset_results",
            )

            st.caption(
                f"Model used: {st.session_state['dataset_model']}. "
                "Predictions are estimates, not verified labels. "
                "Rule-based signals are separate from the model."
            )


# ==================== SINGLE MESSAGE ANALYZER ====================

with tab_single:
    st.subheader("Analyze one message")

    subject = st.text_input("Subject (optional)", key="single_subject")

    body = st.text_area(
        "Message body",
        height=160,
        placeholder="Paste a message here...",
        key="single_body",
    )

    if st.button(
        "Analyze Message",
        type="primary",
        key="analyze_single_message",
    ):
        text = "\n".join(
            part for part in [subject.strip(), body.strip()] if part
        )

        if not text:
            st.warning("Enter a subject or message first.")
        else:
            model = models[best_name]
            prediction = model.predict([text])[0]

            classes = list(model.classes_)
            probabilities = model.predict_proba([text])[0]
            spam_score = float(
                probabilities[classes.index("spam")]
            )

            if prediction == "spam":
                st.error("🚨 Likely Spam")
            else:
                st.success("✅ Likely Legitimate (Ham)")

            st.metric(
                "Model-estimated spam probability",
                f"{spam_score:.1%}",
            )
            st.progress(spam_score)

            st.write(f"**Model used:** {best_name}")
            st.markdown("#### Suspicious patterns")

            signals = explain_message(text)

            if signals:
                for signal in signals:
                    st.warning(signal)
            else:
                st.info(
                    "No rule-based suspicious patterns detected. "
                    "This does not guarantee that the message is safe."
                )

            st.caption(
                "Spam probability is a model estimate and may not "
                "be calibrated. Rule-based signals are not proof of fraud."
            )


# ==================== MODEL EVALUATION ====================

with tab_evaluation:
    st.subheader("Compare trained models")

    st.write(
        f"Both models were evaluated on the same held-out test set "
        f"of {bundle['test_size']:,} messages."
    )

    comparison = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1"]
    ].rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Spam Precision",
            "recall": "Spam Recall",
            "f1": "Spam F1",
        }
    )

    st.dataframe(
        comparison.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("#### Performance comparison")
    st.bar_chart(comparison)

    model_names = list(models.keys())

    selected = st.selectbox(
        "Select a model for its confusion matrix",
        model_names,
        index=model_names.index(best_name),
        key="evaluation_model",
    )

    matrix = np.asarray(
        results[selected]["confusion_matrix"]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Ham", "Spam"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(f"{selected} - Confusion Matrix")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    counts = results[selected]

    st.markdown("#### Prediction errors")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correct Ham", counts["tn"])
    c2.metric("False Positives", counts["fp"])
    c3.metric("Missed Spam", counts["fn"])
    c4.metric("Correct Spam", counts["tp"])

    st.write(
        "**False positives:** Legitimate messages incorrectly "
        "classified as spam."
    )
    st.write(
        "**Missed spam:** Spam messages incorrectly classified "
        "as legitimate."
    )
    st.write(
        "**Model selection:** The selected model is chosen by "
        "spam F1 score rather than accuracy alone."
    )

    st.caption(
        "Evaluation metrics reflect the SMS test dataset only. "
        "Performance may differ on new messages and real emails."
    )
