"""
Credit Risk Prediction Dashboard
German Credit Dataset Analysis
Author: Burak Aktaş
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

# Page config
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .good-score {
        color: #27ae60;
        font-size: 2rem;
        font-weight: bold;
    }
    .bad-score {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('german_credit_risk.csv', index_col=0)
    df_scored = pd.read_csv('german_credit_scored.csv')
    return df, df_scored

@st.cache_resource
def load_model():
    with open('credit_risk_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

df, df_scored = load_data()
model_artifacts = load_model()

# Header
st.markdown('<h1 class="main-header">🏦 Credit Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray;">Machine Learning Model for Banking Credit Assessment</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Dashboard", "🔮 Risk Predictor", "📈 Model Performance", "📋 Data Explorer"])

# ==================== DASHBOARD PAGE ====================
if page == "📊 Dashboard":
    st.header("📊 Overview Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        default_rate = (df['Risk'] == 'bad').mean() * 100
        st.metric(
            label="Default Rate",
            value=f"{default_rate:.1f}%",
            delta=None
        )
    
    with col3:
        avg_credit = df['Credit amount'].mean()
        st.metric(
            label="Avg Credit Amount",
            value=f"€{avg_credit:,.0f}",
            delta=None
        )
    
    with col4:
        avg_duration = df['Duration'].mean()
        st.metric(
            label="Avg Duration",
            value=f"{avg_duration:.0f} months",
            delta=None
        )
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        risk_counts = df['Risk'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=['Good (Paid)', 'Bad (Default)'],
            color_discrete_sequence=['#27ae60', '#e74c3c'],
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution by Risk")
        fig = px.histogram(
            df, x='Age', color='Risk',
            color_discrete_map={'good': '#27ae60', 'bad': '#e74c3c'},
            barmode='overlay', opacity=0.7,
            labels={'Risk': 'Credit Risk'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Default Rate by Purpose")
        purpose_risk = df.groupby('Purpose')['Risk'].apply(lambda x: (x=='bad').mean()*100).sort_values(ascending=True)
        fig = px.bar(
            x=purpose_risk.values,
            y=purpose_risk.index,
            orientation='h',
            color=purpose_risk.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            height=400,
            xaxis_title="Default Rate (%)",
            yaxis_title="Purpose",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Credit Amount vs Duration")
        fig = px.scatter(
            df, x='Duration', y='Credit amount', color='Risk',
            color_discrete_map={'good': '#27ae60', 'bad': '#e74c3c'},
            opacity=0.6, labels={'Risk': 'Credit Risk'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Credit Score Distribution
    st.subheader("Credit Score Distribution")
    fig = px.histogram(
        df_scored, x='Credit_Score', nbins=40,
        color_discrete_sequence=['#3498db']
    )
    fig.add_vline(x=750, line_dash="dash", line_color="#27ae60", annotation_text="Excellent")
    fig.add_vline(x=650, line_dash="dash", line_color="#f39c12", annotation_text="Good")
    fig.add_vline(x=550, line_dash="dash", line_color="#e67e22", annotation_text="Fair")
    fig.add_vline(x=450, line_dash="dash", line_color="#e74c3c", annotation_text="Poor")
    fig.update_layout(height=350, xaxis_title="Credit Score", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# ==================== RISK PREDICTOR PAGE ====================
elif page == "🔮 Risk Predictor":
    st.header("🔮 Credit Risk Predictor")
    st.markdown("Enter customer details to predict credit risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 75, 35)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job", [0, 1, 2, 3], format_func=lambda x: ["Unskilled Non-Resident", "Unskilled Resident", "Skilled", "Highly Skilled"][x])
    
    with col2:
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
    
    with col3:
        credit_amount = st.number_input("Credit Amount (€)", min_value=250, max_value=20000, value=3000)
        duration = st.slider("Duration (months)", 4, 72, 24)
        purpose = st.selectbox("Purpose", df['Purpose'].unique())
    
    if st.button("🔮 Predict Risk", type="primary"):
        # Prepare input
        model = model_artifacts['model']
        label_encoders = model_artifacts['label_encoders']
        
        # Encode inputs
        sex_encoded = label_encoders['Sex'].transform([sex])[0]
        housing_encoded = label_encoders['Housing'].transform([housing])[0]
        saving_encoded = label_encoders['Saving accounts'].transform([saving_accounts])[0]
        checking_encoded = label_encoders['Checking account'].transform([checking_account])[0]
        purpose_encoded = label_encoders['Purpose'].transform([purpose])[0]
        credit_per_month = credit_amount / duration
        
        # Create feature array
        features = np.array([[age, job, credit_amount, duration, credit_per_month,
                            sex_encoded, housing_encoded, saving_encoded,
                            checking_encoded, purpose_encoded]])
        
        # Predict
        risk_prob = model.predict_proba(features)[0][1]
        credit_score = int((1 - risk_prob) * 1000)
        
        # Determine category
        if credit_score >= 750:
            category = "Excellent"
            color = "#27ae60"
            emoji = "🌟"
        elif credit_score >= 650:
            category = "Good"
            color = "#2ecc71"
            emoji = "✅"
        elif credit_score >= 550:
            category = "Fair"
            color = "#f39c12"
            emoji = "⚠️"
        elif credit_score >= 450:
            category = "Poor"
            color = "#e67e22"
            emoji = "🔶"
        else:
            category = "Very Poor"
            color = "#e74c3c"
            emoji = "❌"
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {color}20; border-radius: 10px; padding: 20px; text-align: center;">
                <h3>Credit Score</h3>
                <p style="font-size: 3rem; font-weight: bold; color: {color};">{credit_score}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: {color}20; border-radius: 10px; padding: 20px; text-align: center;">
                <h3>Risk Category</h3>
                <p style="font-size: 2rem; font-weight: bold; color: {color};">{emoji} {category}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: #f0f2f620; border-radius: 10px; padding: 20px; text-align: center;">
                <h3>Default Probability</h3>
                <p style="font-size: 2rem; font-weight: bold; color: {'#e74c3c' if risk_prob > 0.5 else '#27ae60'};">{risk_prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Score"},
            gauge={
                'axis': {'range': [0, 1000]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 450], 'color': "#ffebee"},
                    {'range': [450, 550], 'color': "#fff3e0"},
                    {'range': [550, 650], 'color': "#fffde7"},
                    {'range': [650, 750], 'color': "#e8f5e9"},
                    {'range': [750, 1000], 'color': "#c8e6c9"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': credit_score
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.subheader("📋 Recommendation")
        if category == "Excellent":
            st.success("✅ **Approve** - This customer has an excellent credit profile. Low risk of default.")
        elif category == "Good":
            st.success("✅ **Approve** - This customer has a good credit profile. Acceptable risk level.")
        elif category == "Fair":
            st.warning("⚠️ **Review** - This customer has a fair credit profile. Consider additional documentation or collateral.")
        elif category == "Poor":
            st.warning("🔶 **Caution** - This customer has a poor credit profile. Higher interest rate or collateral recommended.")
        else:
            st.error("❌ **High Risk** - This customer has a very poor credit profile. Consider rejection or strict conditions.")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance")
    
    # Load results
    results_df = pd.read_csv('model_comparison_results.csv')
    
    st.subheader("Model Comparison")
    
    # Metrics table
    st.dataframe(
        results_df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}',
            'AUC-ROC': '{:.4f}'
        }).background_gradient(subset=['AUC-ROC'], cmap='Greens'),
        use_container_width=True
    )
    
    # Bar chart comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            results_df, x='Model', y='AUC-ROC',
            color='AUC-ROC', color_continuous_scale='Greens',
            title='AUC-ROC Score by Model'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            results_df, x='Model', y=['Precision', 'Recall', 'F1-Score'],
            barmode='group', title='Precision, Recall & F1-Score'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    
    model = model_artifacts['model']
    feature_cols = model_artifacts['feature_cols']
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df, x='Importance', y='Feature',
        orientation='h', color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("📌 Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Top Risk Factors:**
        1. Checking Account Status
        2. Credit Amount
        3. Duration of Loan
        4. Age
        5. Purpose of Loan
        """)
    
    with col2:
        st.info("""
        **Model Performance:**
        - Best Model: Random Forest
        - AUC-ROC: 0.7585
        - Accuracy: 72.5%
        - Balanced precision and recall
        """)

# ==================== DATA EXPLORER PAGE ====================
elif page == "📋 Data Explorer":
    st.header("📋 Data Explorer")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect("Risk", df['Risk'].unique(), default=df['Risk'].unique())
    
    with col2:
        purpose_filter = st.multiselect("Purpose", df['Purpose'].unique(), default=df['Purpose'].unique())
    
    with col3:
        age_range = st.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    
    # Filter data
    filtered_df = df[
        (df['Risk'].isin(risk_filter)) &
        (df['Purpose'].isin(purpose_filter)) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1])
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
    
    # Display data
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_credit_data.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: gray;">
    Credit Risk Prediction System | Built with Streamlit & Scikit-learn<br>
    <strong>Author:</strong> Burak Aktaş | <strong>Dataset:</strong> German Credit Risk
</p>
""", unsafe_allow_html=True)
