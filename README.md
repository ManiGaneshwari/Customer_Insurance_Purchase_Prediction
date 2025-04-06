# Customer_Insurance_Purchase_Prediction


## Abstract
This project predicts whether potential customers of a bank-affiliated insurance firm will purchase insurance based on their age and estimated income. Several classification algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Trees, and Random Forest—were applied to an anonymized dataset. After preprocessing and evaluating the models using metrics such as accuracy, precision, recall, and F1-score, Random Forest emerged as the best-performing model. The findings highlight that estimated salary has a stronger influence on insurance purchasing decisions than age, providing valuable insights for targeted marketing strategies.

## Introduction
Understanding the factors that drive customers to purchase insurance products is critical for financial institutions to optimize their offerings and marketing efforts. This project explores the use of artificial intelligence to predict insurance buying behavior using demographic and financial data. By analyzing customer patterns, the research seeks to support data-driven decision-making and enhance overall business performance.

## Literature Review
Previous studies on customer behavior prediction have extensively explored classification models. Logistic Regression remains popular for its simplicity and interpretability (Hosmer & Lemeshow, 2000). KNN and SVM excel at capturing non-linear relationships but face scalability challenges with large datasets (Cortes & Vapnik, 1995). Decision Trees offer high interpretability, while Random Forest, an ensemble method, improves stability and reduces overfitting (Breiman, 2001). Common challenges in applying AI to insurance data include imbalanced datasets and noise. This project builds on existing methodologies by conducting a comparative analysis of these algorithms on a specific insurance dataset.

## Problem Statement
The primary objective is to implement and compare classification algorithms—Logistic Regression, KNN, SVM, Decision Trees, and Random Forest—to identify the most effective model for predicting insurance purchases. The goal is to evaluate each algorithm's performance, extract actionable insights, and select a model that balances accuracy, precision, and generalization while avoiding overfitting.

## Data Collection and Preprocessing
- **Data Preparation**: The dataset was preprocessed and cleaned to ensure consistency and readiness for training. Transformations and feature engineering were applied as needed.
- **Dataset**: The anonymized dataset (`Social_Network_Ads.csv`) contains customer age, estimated salary, and a binary target variable indicating insurance purchase (0 = Not Purchased, 1 = Purchased).

## Methodology
1. **Algorithms Employed**: Logistic Regression, KNN, SVM, Decision Trees, Random Forest.
2. **Metrics Used**: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
3. **Hyperparameter Tuning**: Conducted using cross-validation to optimize model performance.
4. **Selection Criterion**: The top-performing algorithm was chosen based on a comparative analysis of evaluation metrics.

## Implementation
- **Tools**: Python, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
- **Process**: 
  - Data was split into training (80%) and testing (20%) sets.
  - Features (age and salary) were standardized using `StandardScaler`.
  - Models were trained, evaluated, and visualized.
  - Random Forest was implemented with a decision boundary plot to illustrate its performance on the test set.
- **Code**: The Jupyter Notebook (`Customer_Insurance_Purchases.ipynb`) contains the full implementation, including data preprocessing, model training, evaluation, and visualization.

### Example Output
The Random Forest model was used to predict outcomes for new cases:

Age: 30, Salary: 87000 --> Prediction: Purchased

Age: 40, Salary: 0 --> Prediction: Purchased

Age: 40, Salary: 100000 --> Prediction: Purchased

Age: 50, Salary: 0 --> Prediction: Purchased

Age: 18, Salary: 0 --> Prediction: Purchased

Age: 35, Salary: 2500000 --> Prediction: Purchased

Age: 60, Salary: 100000000 --> Prediction: Purchased



## Results
- **Performance**: Random Forest outperformed other models in terms of accuracy and robustness across metrics.
- **Key Insight**: Estimated salary has a greater impact on insurance purchasing decisions than age, as evidenced by graphical analysis (decision boundary plot).
- **Visualization**: A contour plot of the Random Forest decision boundary on the test set highlights the separation between "Purchased" and "Not Purchased" classes.

## How to Run
1. **Prerequisites**:
   - Python 3.x
   - Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
   - Install dependencies: `pip install -r requirements.txt`
2. **Steps**:
   - Clone the repository: `git clone https://github.com/yourusername/customer-insurance-purchases.git`
   - Navigate to the project directory: `cd customer-insurance-purchases`
   - Open the Jupyter Notebook: `jupyter notebook Customer_Insurance_Purchases.ipynb`
   - Run all cells to preprocess data, train models, and visualize results.
3. **Dataset**: Ensure `Social_Network_Ads.csv` is in the project directory (included in the repository).

## GitHub Repository
[Customer Insurance Purchases Project](https://github.com/ManiGaneshwari/Customer_Insurance_Purchase_Prediction)

## Future Work
- Incorporate additional features (e.g., gender, occupation) to improve prediction accuracy.
- Address potential class imbalance in the dataset using techniques like SMOTE.
- Explore deep learning models for comparison with traditional algorithms.

## References
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273-297.
- Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression*. Wiley.
