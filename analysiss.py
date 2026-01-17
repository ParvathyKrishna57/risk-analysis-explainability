from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

class RiskAnalyzer:
    """
    Handles student learning risk analysis and explainability - Member 4 GSoC Deliverable
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def get_feature_importance(self, model) -> pd.DataFrame:
        """
        Extract feature importance from tree-based ML models (Random Forest)
        """
        importance = model.feature_importances_
        return pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

    def rule_based_risk_analysis(self, student_data: Dict) -> str:
        """
        Generate clear, human-readable explanation for student risk
        """
        risk_factors = []

        if student_data.get("attendance", 100) < 75:
            risk_factors.append("low attendance")
        if student_data.get("assignment_completion", 100) < 60:
            risk_factors.append("poor assignment completion")
        if student_data.get("study_hours", 10) < 2:
            risk_factors.append("insufficient study hours")
        if student_data.get("test_score", 100) < 40:
            risk_factors.append("low test performance")

        if not risk_factors:
            return (
                "Risk Level: Low\n"
                "Explanation: No major learning risks detected.\n"
                "Student performance appears stable.\n"
            )

        explanation = (
            f"Risk Level: High\n"
            f"Explanation: "
            f"{' and '.join(risk_factors).capitalize()} "
            f"are the primary contributors to the student's learning risk."
        )
        return explanation

    def plot_importance(self, df_importance: pd.DataFrame, top_n: int = 5) -> None:
        """
        Colorful compact feature importance plot for GSoC README
        """
        top_features = df_importance.head(top_n)
        colors = plt.cm.RdYlGn_r(top_features['Importance'])
        
        plt.figure(figsize=(6, 4))
        bars = plt.barh(top_features['Feature'], top_features['Importance'], 
                        color=colors, edgecolor='black', linewidth=1.2)
        
        # Add % labels on bars
        for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
            plt.text(imp + 0.01, i, f'{imp:.1%}', va='center', fontweight='bold')
        
        plt.title('Top Student Risk Factors', fontsize=14, fontweight='bold', pad=10)
        plt.xlabel('Importance (%)', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # High-quality save for GitHub README
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()

# ------------------ Complete Test & Demo ------------------
if __name__ == "__main__":
    # Sample student data
    sample_student = {
        "attendance": 80,
        "assignment_completion": 82,
        "study_hours": 2.5,
        "test_score": 48
    }

    # Initialize analyzer with your 5 features
    analyzer = RiskAnalyzer([
        "attendance", "study_hours", "assignment_completion", 
        "test_score", "participation"
    ])

    print("🎯 STUDENT RISK ANALYZER ")
    print("=" * 60)
    
    # 1. Test rule-based analysis
    print("\n1️⃣ RULE-BASED RISK ANALYSIS:")
    print(analyzer.rule_based_risk_analysis(sample_student))
    print("\n" + "="*60)

    # 2. Simulate feature importance (replace with Member 3's real model)
    fake_importance = [0.35, 0.25, 0.20, 0.15, 0.05]  # From RandomForest
    
    importance_df = pd.DataFrame({
        "Feature": analyzer.feature_names,
        "Importance": fake_importance
    }).sort_values(by="Importance", ascending=False)
    
    print("\n2️⃣ FEATURE IMPORTANCE (Model Output):")
    print(importance_df.round(3))
    print("\n" + "="*60)

    # 3. Generate professional plot
    print("3️⃣ GENERATING VISUALIZATION...")
    analyzer.plot_importance(importance_df)
    print("✅ feature_importance.png saved for GitHub README!")
