import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import make_pipeline
import numpy as np
import requests
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="Fake News Detection System", layout="wide", page_icon="📰")

# Load NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Helper Functions ---
@st.cache_resource
def load_models():
    models = joblib.load('models.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return models, vectorizer

HISTORY_FILE = 'history.csv'

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=['Timestamp', 'Text', 'Prediction', 'Confidence'])

def save_history(text, prediction, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        'Timestamp': timestamp,
        'Text': text,
        'Prediction': prediction,
        'Confidence': confidence
    }])
    if not os.path.exists(HISTORY_FILE):
        new_entry.to_csv(HISTORY_FILE, index=False)
    else:
        new_entry.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator=' ')
        return text
    except Exception as e:
        return None

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def create_gauge_chart(probability):
    # Probability of being True (0 to 1)
    score = probability * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credibility Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FF4B4B'},  # Red for Fake
                {'range': [50, 100], 'color': '#2ECC71'}  # Green for True
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    # Add label annotation
    label = "LIKELY TRUE" if score >= 50 else "LIKELY FAKE"
    color = "#2ECC71" if score >= 50 else "#FF4B4B"
    
    fig.add_annotation(x=0.5, y=0.25, text=label, showarrow=False, font=dict(size=20, color=color))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_confidence_bar_chart(prob_fake, prob_true):
    data = pd.DataFrame({
        'Label': ['Fake News', 'Real News'],
        'Probability': [prob_fake, prob_true],
        'Color': ['#FF4B4B', '#2ECC71']
    })
    
    fig = px.bar(data, x='Label', y='Probability', color='Label', 
                 color_discrete_map={'Fake News': '#FF4B4B', 'Real News': '#2ECC71'},
                 text_auto='.1%', title="Prediction Confidence")
    
    fig.update_layout(showlegend=False, height=300, yaxis_range=[0, 1])
    return fig

def plot_lime_features(exp_list):
    # exp_list is a list of tuples (word, weight)
    features = [x[0] for x in exp_list]
    weights = [x[1] for x in exp_list]
    
    # Sort by absolute weight for "Feature Importance" view
    abs_weights = [abs(w) for w in weights]
    
    # Create DataFrame
    df = pd.DataFrame({'Word': features, 'Weight': weights, 'AbsWeight': abs_weights})
    df = df.sort_values(by='AbsWeight', ascending=True) # Ascending for horizontal bar chart
    
    # Determine color based on contribution (Red for Fake/Negative, Green for True/Positive)
    # Note: LIME weights: positive = towards class 1 (True), negative = towards class 0 (Fake)
    df['Color'] = df['Weight'].apply(lambda x: '#2ECC71' if x > 0 else '#FF4B4B')
    df['Contribution'] = df['Weight'].apply(lambda x: 'Push toward Real' if x > 0 else 'Push toward Fake')
    
    fig = px.bar(df, x='Weight', y='Word', orientation='h', color='Contribution',
                 color_discrete_map={'Push toward Real': '#2ECC71', 'Push toward Fake': '#FF4B4B'},
                 title="LIME Analysis: Feature Contributions")
    
    fig.update_layout(height=400)
    return fig, df

# --- UI Layout ---
st.title("📰 Fake News Detection System")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Analysis Dashboard", "History", "Project Overview"])

models, vectorizer = load_models()

if page == "Analysis Dashboard":
    st.header("🕵️‍♂️ Analysis Dashboard")
    
    input_method = st.radio("Choose Input Method:", ["Text", "URL"], horizontal=True)
    
    news_text = ""
    
    if input_method == "Text":
        news_text = st.text_area("Paste the news article text here:", height=150, placeholder="Enter news content...")
    else:
        url_input = st.text_input("Enter Article URL:", placeholder="https://example.com/article")
        if url_input:
            with st.spinner("Fetching content from URL..."):
                fetched_text = get_text_from_url(url_input)
                if fetched_text:
                    news_text = fetched_text
                    st.success("Content fetched successfully!")
                    with st.expander("View Fetched Text"):
                        st.write(news_text[:1000] + "..." if len(news_text) > 1000 else news_text)
                else:
                    st.error("Failed to fetch content from the URL. Please check the URL or try pasting the text manually.")

    if st.button("Analyze News", type="primary"):
        if news_text:
            with st.spinner("Processing and Analyzing..."):
                # Preprocess
                cleaned_text = clean_text(news_text)
                vectorized_text = vectorizer.transform([cleaned_text])
                
                # 1. Main Prediction (Using Logistic Regression as primary/default)
                primary_model = models['Logistic Regression']
                prob = primary_model.predict_proba(vectorized_text)[0]
                prob_fake, prob_true = prob[0], prob[1]
                
                # Save to History
                prediction_label = "Real News" if prob_true > 0.5 else "Fake News"
                confidence_score = prob_true if prob_true > 0.5 else prob_fake
                save_history(news_text, prediction_label, confidence_score)
                
                # --- Section 1: Analysis Results ---
                st.subheader("📊 Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_gauge_chart(prob_true), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_confidence_bar_chart(prob_fake, prob_true), use_container_width=True)
                
                # --- Section 2: Explainability ---
                st.subheader("💡 Explainability: What Influenced This Prediction?")
                
                # Generate LIME explanation
                c = make_pipeline(vectorizer, primary_model)
                explainer = LimeTextExplainer(class_names=['Fake', 'True'])
                exp = explainer.explain_instance(cleaned_text, c.predict_proba, num_features=10)
                
                tab1, tab2 = st.tabs(["🔍 LIME Analysis", "📝 Key Influential Words"])
                
                lime_fig, lime_df = plot_lime_features(exp.as_list())
                
                with tab1:
                    st.plotly_chart(lime_fig, use_container_width=True)
                    st.caption("Green bars indicate words contributing to 'Real News', Red bars indicate words contributing to 'Fake News'.")
                
                with tab2:
                    st.markdown("### Top Influential Words")
                    # Display as a nice list or table
                    top_words = lime_df.sort_values(by='AbsWeight', ascending=False).head(5)
                    for index, row in top_words.iterrows():
                        color = "green" if row['Weight'] > 0 else "red"
                        st.markdown(f"- **{row['Word']}**: <span style='color:{color}'>{row['Weight']:.4f}</span>", unsafe_allow_html=True)

                # --- Section 3: Advanced Analysis (Model Comparison) ---
                st.subheader("🔬 Advanced Analysis")
                
                # Get predictions from all models
                model_results = []
                for name, model in models.items():
                    p = model.predict_proba(vectorized_text)[0]
                    pred_label = "Real News" if p[1] > 0.5 else "Fake News"
                    conf = p[1] if p[1] > 0.5 else p[0]
                    model_results.append({
                        'Model': name,
                        'Prediction': pred_label,
                        'Confidence': conf,
                        'Prob_Real': p[1],
                        'Prob_Fake': p[0]
                    })
                
                results_df = pd.DataFrame(model_results)
                
                # Grouped Bar Chart
                st.markdown("#### Model Comparison")
                
                # Prepare data for grouped bar chart
                comparison_data = []
                for res in model_results:
                    comparison_data.append({'Model': res['Model'], 'Type': 'Real News', 'Probability': res['Prob_Real']})
                    comparison_data.append({'Model': res['Model'], 'Type': 'Fake News', 'Probability': res['Prob_Fake']})
                
                comp_df = pd.DataFrame(comparison_data)
                
                fig_comp = px.bar(comp_df, x='Model', y='Probability', color='Type', barmode='group',
                                  color_discrete_map={'Real News': '#2ECC71', 'Fake News': '#FF4B4B'},
                                  text_auto='.2f')
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Detailed Table
                st.markdown("#### Detailed Model Results")
                st.table(results_df[['Model', 'Prediction', 'Confidence']].style.format({'Confidence': '{:.2%}'}))
                
                # Footer
                st.info(f"Model Used for Main Prediction: Logistic Regression")
                st.warning("⚠️ This is a demonstration system. Always verify news from multiple reliable sources.")
                    
        else:
            st.warning("Please enter some text to analyze.")

elif page == "History":
    st.header("📜 Analysis History")
    df = load_history()
    if not df.empty:
        # Display as a dataframe
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name='fake_news_history.csv',
            mime='text/csv',
        )
        
        if st.button("Clear History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
                st.success("History cleared!")
                st.rerun()
    else:
        st.info("No history found. Analyze some news to see it here!")

elif page == "Project Overview":
    st.header("ℹ️ Project Overview")
    st.markdown("""
    ### About Fake News
    Fake news consists of deliberate disinformation or hoaxes spread via traditional news media or online social media.
    
    ### Dataset Information
    - **True News**: Articles from reliable sources.
    - **Fake News**: Articles identified as unreliable or fabricated.
    
    ### Models Used
    1. **Logistic Regression**: A statistical model that uses a logistic function to model a binary dependent variable.
    2. **Random Forest**: An ensemble learning method operating by constructing a multitude of decision trees.
    3. **Support Vector Machine (SVM)**: A supervised learning model that analyzes data for classification and regression analysis.
    """)

