import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.decomposition import PCA
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
import io
import base64


sns.set_style("whitegrid")

class FairHiringDashboard:
    def __init__(self, file_path):
        """Initialize the dashboard with dataset and configurations."""
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}
        self.pca = PCA(n_components=2)
        self.models = {
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Fair Logistic (DP)": None,  # Demographic Parity
            "Fair Logistic (EO)": None   # Equalized Odds
        }
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.sensitive_train, self.sensitive_test = None, None

    def load_data(self):
        """Phase 1: Load and preprocess the dataset."""
        try:
            self.df = pd.read_csv(self.file_path)
            self.preprocess_data()
        except FileNotFoundError:
            st.error("Dataset 'hiring_data.csv' not found. Please place it in the project directory.")
            st.stop()

    def preprocess_data(self):
        """Phase 1: Data Preprocessing & Cleaning"""
        # Handle missing values
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        self.df.fillna(self.df.mode().iloc[0], inplace=True)

        
        numeric_cols = ['Age', 'ExperienceYears', 'DistanceFromCompany', 
                        'InterviewScore', 'SkillScore', 'PersonalityScore']
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df[col] = self.df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        
        categorical_cols = ['Gender', 'Qualification', 'RecruitmentStrategy', 'HiringDecision']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

    
        corr = self.df.corr(numeric_only=True)['HiringDecision'].abs()
        self.features = corr[corr > 0.1].index.drop(['HiringDecision', 'EmpID', 'Name'], errors='ignore').tolist()

        
        self.sensitive_attrs = ['Gender', 'Age']
        self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=[0, 30, 40, 100], labels=['<30', '30-40', '>40'])

        
        self.df = self.df.drop(columns=['EmpID', 'Name'])

    def exploratory_data_analysis(self):
        """Phase 2: Exploratory Data Analysis (EDA)"""
        st.header("Exploratory Data Analysis")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            gender_filter = st.multiselect("Filter by Gender", options=['Male', 'Female'], default=['Male', 'Female'])
        with col2:
            age_filter = st.multiselect("Filter by Age Group", options=['<30', '30-40', '>40'], default=['<30', '30-40', '>40'])

        filtered_df = self.df[
            (self.df['Gender'].map(lambda x: self.label_encoders['Gender'].inverse_transform([x])[0]).isin(gender_filter)) &
            (self.df['AgeGroup'].isin(age_filter))
        ]

        # Pie Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gender Distribution")
            fig, ax = plt.subplots()
            filtered_df['Gender'].value_counts().rename(index=lambda x: self.label_encoders['Gender'].inverse_transform([x])[0]).plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Hiring Decisions")
            fig, ax = plt.subplots()
            filtered_df['HiringDecision'].value_counts().rename(index={0: 'Not Hired', 1: 'Hired'}).plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

        # Bar Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hiring by Experience Years")
            fig, ax = plt.subplots()
            sns.barplot(x='ExperienceYears', y='HiringDecision', data=filtered_df, ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Hiring by Recruitment Strategy")
            fig, ax = plt.subplots()
            sns.barplot(x='RecruitmentStrategy', y='HiringDecision', data=filtered_df, ax=ax)
            st.pyplot(fig)

        # Histograms
        st.subheader("Score Distributions")
        fig, ax = plt.subplots()
        for col in ['InterviewScore', 'SkillScore', 'PersonalityScore']:
            sns.histplot(filtered_df[col], label=col, ax=ax, bins=20, alpha=0.5)
        ax.legend()
        st.pyplot(fig)

        # Heatmap
        st.subheader("Feature Correlations")
        fig, ax = plt.subplots()
        sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Bias Check
        st.subheader("Bias Check Across Sensitive Groups")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.barplot(x='Gender', y='HiringDecision', data=filtered_df, ax=ax)
            ax.set_xticklabels([self.label_encoders['Gender'].inverse_transform([int(x.get_text())])[0] for x in ax.get_xticklabels()])
            ax.set_title("Hiring Rate by Gender")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.barplot(x='AgeGroup', y='HiringDecision', data=filtered_df, ax=ax)
            ax.set_title("Hiring Rate by Age Group")
            st.pyplot(fig)

    def train_models(self):
        """Phase 3: Train Machine Learning Models"""
        X = self.df[self.features]
        y = self.df['HiringDecision']
        sensitive_features = self.df['Gender']

        self.X_train, self.X_test, self.y_train, self.y_test, self.sensitive_train, self.sensitive_test = train_test_split(
            X, y, sensitive_features, test_size=0.2, random_state=42
        )

        # Train standard models
        for name in ["KNN", "Random Forest", "Logistic Regression"]:
            self.models[name].fit(self.X_train, self.y_train)

        # Train fairness-aware models
        fair_dp = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=DemographicParity(), max_iter=50)
        fair_dp.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_train)
        self.models["Fair Logistic (DP)"] = fair_dp

        fair_eo = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=EqualizedOdds(), max_iter=50)
        fair_eo.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_train)
        self.models["Fair Logistic (EO)"] = fair_eo

    def fairness_analysis(self):
        """Fairness Metrics Visualization"""
        st.header("Fairness Analysis")
        model_choice = st.selectbox("Select Model for Fairness Analysis", list(self.models.keys()))
        sensitive_attr = st.selectbox("Select Sensitive Attribute", ["Gender", "AgeGroup"])

        y_pred = self.models[model_choice].predict(self.X_test)
        mf = MetricFrame(
            metrics={
                'Selection Rate': selection_rate,
                'True Positive Rate': true_positive_rate,
                'False Positive Rate': false_positive_rate
            },
            y_true=self.y_test,
            y_pred=y_pred,
            sensitive_features=self.df.loc[self.X_test.index, sensitive_attr]
        )

        st.write(f"Fairness Metrics by {sensitive_attr}:")
        st.dataframe(mf.by_group)

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        mf.by_group.plot.bar(subplots=True, layout=(3, 1), ax=ax, legend=False)
        plt.tight_layout()
        st.pyplot(fig)

    def model_evaluation(self):
        """Phase 5: Model Evaluation"""
        st.header("Model Evaluation")
        self.train_models()  

        evaluations = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            evaluations[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'R2': r2_score(self.y_test, y_pred),
                'Confusion Matrix': confusion_matrix(self.y_test, y_pred)
            }

        # Display metrics
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Model': evaluations.keys(),
            'Accuracy': [m['Accuracy'] for m in evaluations.values()],
            'RMSE': [m['RMSE'] for m in evaluations.values()],
            'R2': [m['R2'] for m in evaluations.values()]
        })
        st.table(metrics_df.style.format("{:.2f}", subset=['Accuracy', 'RMSE', 'R2']))

        # Confusion Matrices
        st.subheader("Confusion Matrices")
        cols = st.columns(len(self.models))
        for idx, (name, metrics) in enumerate(evaluations.items()):
            with cols[idx]:
                st.write(f"{name}")
                fig, ax = plt.subplots(figsize=(3, 3))
                sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

        # Fairness 
        st.subheader("Performance vs Fairness Tradeoff")
        fig, ax = plt.subplots()
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            mf = MetricFrame(metrics=selection_rate, y_true=self.y_test, y_pred=y_pred, 
                            sensitive_features=self.df.loc[self.X_test.index, 'Gender'])
            sr_diff = mf.by_group.max() - mf.by_group.min()
            acc = evaluations[name]['Accuracy']
            plt.scatter(sr_diff, acc, label=name)
        plt.xlabel("Selection Rate Difference (Gender)")
        plt.ylabel("Accuracy")
        plt.legend()
        st.pyplot(fig)

    def pca_analysis(self):
        """Phase 4: Dimensionality Reduction (PCA)"""
        st.header("PCA Analysis")
        X = self.df[self.features]
        X_pca = self.pca.fit_transform(X)
        explained_variance = self.pca.explained_variance_ratio_

        st.write(f"Explained Variance Ratio: {explained_variance}")
        fig, ax = plt.subplots()
        plt.bar(range(len(explained_variance)), explained_variance)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance")
        st.pyplot(fig)

        
        pca_model = LogisticRegression(max_iter=1000)
        pca_model.fit(X_pca[self.X_train.index], self.y_train)
        pca_accuracy = accuracy_score(self.y_test, pca_model.predict(X_pca[self.X_test.index]))
        st.write(f"Accuracy with PCA features: {pca_accuracy:.2f}")

        
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=self.df['HiringDecision'].map({0: 'Not Hired', 1: 'Hired'}), ax=ax)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        st.pyplot(fig)

    def live_prediction(self):
        """Interactive Prediction with Fairness Toggle"""
        st.header("Live Hiring Prediction")
        model_choice = st.selectbox("Select Model", list(self.models.keys()))
        use_ai_mode = st.checkbox("Enable AI Fairness Mode", value=False, help="Uses fairness-aware model for equitable predictions")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 20, 50, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                exp_years = st.slider("Experience Years", 0, 15, 5)
                distance = st.slider("Distance from Company (km)", 0.0, 50.0, 10.0)
            with col2:
                interview = st.slider("Interview Score", 0, 100, 50)
                skill = st.slider("Skill Score", 0, 100, 50)
                personality = st.slider("Personality Score", 0, 100, 50)
                strategy = st.selectbox("Recruitment Strategy", [1, 2, 3])
            submit = st.form_submit_button("Predict")

        if submit:
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [self.label_encoders['Gender'].transform([gender])[0]],
                'ExperienceYears': [exp_years],
                'DistanceFromCompany': [distance],
                'InterviewScore': [interview],
                'SkillScore': [skill],
                'PersonalityScore': [personality],
                'RecruitmentStrategy': [strategy-1]
            }, columns=self.features)
            model_key = "Fair Logistic (DP)" if use_ai_mode else model_choice
            prediction = self.models[model_key].predict(input_data)[0]
            st.success(f"Prediction: {'Hired' if prediction == 1 else 'Not Hired'}")
            if use_ai_mode:
                st.info("AI Fairness Mode: Prediction adjusted to minimize bias (Demographic Parity).")

    def download_report(self):
        """Generate and offer a downloadable report"""
        buffer = io.StringIO()
        evaluations = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            mf = MetricFrame(metrics=selection_rate, y_true=self.y_test, y_pred=y_pred, 
                            sensitive_features=self.df.loc[self.X_test.index, 'Gender'])
            evaluations[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Selection Rate Diff': mf.by_group.max() - mf.by_group.min()
            }
        report_df = pd.DataFrame.from_dict(evaluations, orient='index')
        report_df.to_csv(buffer)
        buffer.seek(0)
        st.download_button(
            label="Download Fairness Report",
            data=buffer.getvalue(),
            file_name="fairness_report.csv",
            mime="text/csv"
        )

    def run(self):
        """Main method to run the dashboard"""
        st.set_page_config(page_title="Fair Hiring Dashboard", layout="wide")
        st.title("Fair Hiring Decision Dashboard")
        st.markdown("Analyze hiring decisions with fairness-aware models to ensure equitable outcomes.")

    
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["EDA", "Fairness Analysis", "Model Evaluation", "PCA", "Prediction"])
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.write("Built with fairness-aware AI to minimize bias in recruitment systems.")
        st.sidebar.markdown("---")

        
        self.load_data()
        self.train_models()  
        self.download_report()

        
        if page == "EDA":
            self.exploratory_data_analysis()
        elif page == "Fairness Analysis":
            self.fairness_analysis()
        elif page == "Model Evaluation":
            self.model_evaluation()
        elif page == "PCA":
            self.pca_analysis()
        elif page == "Prediction":
            self.live_prediction()


if __name__ == "__main__":
    dashboard = FairHiringDashboard("hiring_data.csv")
    dashboard.run()