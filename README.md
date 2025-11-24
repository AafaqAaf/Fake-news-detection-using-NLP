# 📰 Fake News Detection System

A comprehensive Machine Learning system designed to detect fake news articles using Natural Language Processing (NLP). This project includes a Jupyter Notebook for the full data science workflow and a Streamlit web application for real-time inference with explainability.

## 🚀 Features

-   **Multi-Model Classification**: Utilizes Logistic Regression, Random Forest, and SVM for robust predictions.
-   **Interactive Web App**: Built with Streamlit, featuring a modern UI with:
    -   **Credibility Gauge**: Visual representation of the "True" probability.
    -   **Confidence Charts**: Detailed breakdown of model confidence.
    -   **LIME Explainability**: Highlights specific words that influenced the prediction (Green for True, Red for Fake).
    -   **Model Comparison**: Side-by-side performance metrics of all three models.
-   **Full Pipeline Notebook**: A Jupyter Notebook covering data loading, cleaning, EDA, training, and evaluation.

## 📂 Project Structure

```
├── app.py                      # Streamlit Web Application
├── train.py                    # Script to train models and save artifacts
├── Fake_News_Detection.ipynb   # Jupyter Notebook with full analysis
├── models.joblib               # Saved trained models (LR, RF, SVM)
├── vectorizer.joblib           # Saved TF-IDF Vectorizer
├── True.csv                    # Dataset containing real news
├── Fake.csv                    # Dataset containing fake news
└── README.md                   # Project Documentation
```

## 🛠️ Installation

1.  **Clone the repository** (or download the files).
2.  **Install dependencies**:
    Ensure you have Python installed. Install the required libraries using pip:

    ```bash
    pip install pandas numpy scikit-learn nltk beautifulsoup4 plotly streamlit joblib matplotlib seaborn lime
    ```

## 🏃‍♂️ How to Run

### 1. Run the Web App (Quick Start)
If the model files (`models.joblib` and `vectorizer.joblib`) are already present, you can run the app immediately:

```bash
streamlit run app.py
```

### 2. Retrain Models (Optional)
If you want to retrain the models or if the artifact files are missing, run the training script:

```bash
python train.py
```
This will generate new `models.joblib` and `vectorizer.joblib` files.

### 3. Explore the Notebook
Open `Fake_News_Detection.ipynb` in Jupyter Notebook or VS Code to explore the data analysis, visualizations, and training process step-by-step.

## 📊 Model Performance

The system evaluates three models:
-   **Logistic Regression**: Fast and interpretable (Used for LIME explanations).
-   **Random Forest**: High accuracy, robust against overfitting.
-   **SVM (Linear)**: Excellent performance on high-dimensional text data.

*Detailed performance metrics (Accuracy, Confusion Matrix, ROC Curves) can be found in the Jupyter Notebook.*

## 📝 Dataset
The project uses two datasets:
-   `True.csv`: Articles from reliable sources.
-   `Fake.csv`: Articles identified as unreliable.
*(Ensure these files are in the root directory before running the training script).*
