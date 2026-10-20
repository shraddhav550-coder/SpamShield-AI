 SpamShield AI — Intelligent Spam Detection System

An interactive machine learning application that detects spam messages, analyzes datasets in bulk, and helps users understand suspicious message patterns through an intuitive dashboard.

**Built with Python, Scikit-learn, Pandas, and Streamlit.**

 Overview

Spam messages often contain misleading offers, suspicious links, urgent requests, and attempts to obtain sensitive information. SpamShield AI uses Natural Language Processing (NLP) and machine learning classification algorithms to distinguish spam messages from legitimate messages.

The application provides both single-message detection and bulk dataset analysis, allowing users to classify messages, explore model performance, and export prediction results.

 Key Features

- **  Single Message Detection** — Enter a message and predict whether it is spam or legitimate.
- ** Bulk Dataset Analysis** — Upload a CSV file and classify messages across an entire dataset.
- ** Interactive Dashboard** — View spam counts, legitimate message counts, and classification summaries.
- ** Suspicious Pattern Detection** — Highlight patterns such as suspicious links, urgent language, prize claims, payment requests, and requests for sensitive information.
- ** Model Comparison** — Compare Multinomial Naive Bayes and Logistic Regression.
- ** Model Evaluation** — Review confusion matrices, false positives, false negatives, true positives, and true negatives.
- ** Export Predictions** — Download analyzed messages and their predictions as a CSV file.
- **  Fast Predictions** — Use trained models to classify messages without retraining for every prediction.

 Machine Learning Approach

SpamShield AI follows a text-classification workflow:

1. **Data preparation:** Load and clean the SMS Spam Collection dataset.
2. **Text preprocessing:** Prepare message text for machine learning.
3. **Feature extraction:** Convert text into numerical features using TF-IDF vectorization.
4. **Model training:** Train and compare classification algorithms.
5. **Evaluation:** Measure model performance on a held-out test set.
6. **Model selection:** Select the model based on spam-class F1-score.
7. **Prediction:** Use the saved model to classify new messages through the Streamlit interface.

 Models Compared

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.41% | 100.00% | 71.09% | 83.11% |
| Logistic Regression | **97.77%** | 88.32% | **94.53%** | **91.32%** |

*Results from the project's recorded held-out evaluation. Values are rounded to two decimal places.*

Selected Model: Logistic Regression

Logistic Regression was selected because it achieved the higher spam-class F1-score of **91.32%** and detected a larger proportion of spam messages.

- **Accuracy — 97.77%:** The proportion of all test messages classified correctly.
- **Precision — 88.32%:** The proportion of messages predicted as spam that were actually spam.
- **Recall — 94.53%:** The proportion of actual spam messages successfully detected.
- **F1-score — 91.32%:** The balance between spam precision and recall.

High recall is valuable for spam detection because it helps reduce the number of spam messages incorrectly classified as legitimate.

Dataset

This project uses the **SMS Spam Collection**, a public dataset containing labeled SMS messages.

Each message is categorized as:
- `ham` — legitimate message
- `spam` — spam message

After cleaning, the recorded dataset contained **5,158 messages**:

| Category | Messages |
|---|---:|
| Legitimate (Ham) | 4,516 |
| Spam | 642 |
| **Total** | **5,158** |

**Dataset source:** [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset is intended for SMS spam detection. Predictions on emails or other types of text may be less reliable because those sources can have different language patterns.

 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| Scikit-learn | Text vectorization, model training, and evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | Selected spam classification model |
| Multinomial Naive Bayes | Baseline model comparison |
| Streamlit | Interactive web application |
| Matplotlib | Evaluation visualizations |
| Joblib | Saving and loading trained models |
| Git & GitHub | Version control and project hosting |

 Project Structure

```text
SpamShield-AI/
│
├── app.py                         # Streamlit application
├── train_model.py                 # Model training and evaluation
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files excluded from Git
├── test_messages.csv              # Sample messages for testing
│
├── data/                          # Dataset directory
│
└── models/
    ├── spam_classifier.joblib     # Selected trained model
    └── model_comparison.joblib    # Saved comparison results
```

 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- Git
- Visual Studio Code (recommended)

1. Clone the repository

```bash
git clone https://github.com/shraddhav550-coder/SpamShield-AI.git
cd SpamShield-AI
```

2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

4. Launch the application

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, visit:

`http://localhost:8501`

### 5. Retrain the models (optional)

If you want to retrain the classifiers using the dataset configured in the training script, run:

```bash
python train_model.py
```

Make sure the required dataset is available at the path expected by `train_model.py` before running this command.

 How to Use SpamShield AI

 1. Single Message

Enter a message into the application and run the prediction to view the predicted class, estimated spam probability, and suspicious patterns.

 2. Dataset Analyzer

Upload a CSV file containing a column of message text. Select the relevant column if prompted, analyze the messages, review the summary, filter predictions, and download the results.

Example CSV:

```csv
message
"Congratulations! You have won a free prize. Click here."
"Hey, are we still meeting at 5 PM?"
"URGENT: Claim your reward now!"
```

 3. Model Evaluation

Explore the available model comparison results and confusion matrices to understand how the classifiers perform and where their predictions differ.

 Important Notes

- SpamShield AI is a machine learning demonstration, not a guaranteed spam filter.
- The predicted spam probability is a model estimate, not a guarantee that a message is malicious.
- Suspicious-pattern indicators are rule-based explanations and do not independently establish that a message is spam.
- The current training dataset consists of SMS messages, so performance on email, social media, or other text sources may differ.
- The dataset analyzer classifies uploaded messages; it does not automatically retrain the model on uploaded data.
- The application requires the saved model files in `models/` to be present when loading the existing trained model.

 Future Improvements

- [ ] Add email-specific spam detection datasets.
- [ ] Display precision, recall, F1-score, and accuracy dynamically in the dashboard.
- [ ] Add evaluation against ground-truth labels in uploaded datasets.
- [ ] Improve explanations using interpretable NLP techniques.
- [ ] Add confidence thresholds and adjustable classification settings.
- [ ] Support larger datasets with more efficient processing.
- [ ] Add a persistent database and prediction history.
- [ ] Deploy the application online for public access.
- [ ] Add automated tests and continuous integration.

 Learning Outcomes

This project demonstrates practical experience with:

- Natural Language Processing and text classification
- Data cleaning and exploratory analysis
- Feature extraction using TF-IDF
- Training and comparing supervised machine learning models
- Evaluation using precision, recall, F1-score, and confusion matrices
- Building interactive ML dashboards with Streamlit
- Saving and loading trained models with Joblib
- Managing project versions using Git and GitHub



