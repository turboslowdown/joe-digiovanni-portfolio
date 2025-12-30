# Joe's Portfolio

## About

This portfolio showcases my work in baseball analytics and data science, with a focus on predictive modeling, statistical analysis, and applied research for Major League Baseball organizations. My projects demonstrate expertise in machine learning, statistical modeling, data visualization, and the ability to translate complex analytical insights into actionable recommendations for player evaluation and game strategy.



Contact:

For inquiries about my work or collaboration opportunities, please reach out! I can be reached below:

[linkedin.com/in/josephndigiovanni](https://linkedin.com/in/josephndigiovanni)

jdigiovanni@uchicago.edu

joseph.n.digiovanni@gmail.com


## Portfolio Structure

### Analytical & Applied Projects

- **Catcher Framing Prediction** - A production-ready machine learning model using HistGradientBoostingClassifier to predict the probability that a pitch will be called a strike, enabling evaluation of catcher framing ability. The model incorporates pitch location (normalized to strike zone boundaries), pitch movement (horizontal and vertical break), velocity, pitch type, and batter/pitcher handedness. The production pipeline processes new pitch data, generates pitch-level predictions, and aggregates results to catcher-level metrics including "strikes added" per 100 opportunities. The model uses group-based cross-validation (splitting by game ID) to prevent data leakage and ensure robust performance on unseen games.

- **Predicting Pitch Velocity Differential** - A comprehensive machine learning project predicting the velocity difference between a pitcher's primary fastball (average of four-seam, two-seam, and sinker) and their breaking balls. The analysis includes extensive data cleaning (filtering position players, handling missing values), feature engineering (creating differential features for break, release point, spin rate, and spin axis relative to fastball baseline), and model comparison between Random Forest and XGBoost. The final model achieves R² of 0.94 with MAE of 0.85 MPH, identifying vertical break differential as the dominant predictive factor. The project demonstrates advanced understanding of pitch physics and the relationship between pitch characteristics and velocity separation, with applications for pitch development and deception optimization.

- **Postgame Pitch Analysis App** - Interactive Streamlit dashboard for analyzing pitcher performance from single-game pitch tracking data. Features include:
  - Pitch mix analysis with usage percentages
  - Movement profiles (horizontal vs vertical break)
  - Pitch location plots with strike zone overlay
  - Interactive filtering by outcome, batter handedness, and count situation

- **Next Pitch Prediction Model**: LightGBM model trained on engineered features within Statcast data from the 2024 season aiming to predict the next pitch thrown by a pitcher in an at bat based on the sequencing, historical results, pitcher tendencies, hitter tendencies, and game state. 

### Statistical Modeling & EDA Projects

- **Statistical Models for Data Science** - A mortgage default risk analysis project using logistic regression (logit and probit models) to predict loan defaults within the first 36 months. The analysis merges loan origination data with servicing performance data, creates default and defect flags based on loan status codes, and builds predictive models using credit score, loan-to-value ratios, debt-to-income ratios, loan amount, and interest rate. The project includes model assessment (deviance tests, AIC comparison), confusion matrix analysis, and loss calculation with expected vs. observed loss comparisons. A refined model incorporates interaction terms and categorical variables (loan purpose, occupancy status) to improve predictive performance, with detailed evaluation of accuracy, precision, recall, and profit calculations for loan decision-making.

### Big Data Analysis

- **Big Data Repository Analysis** - A comprehensive PySpark analysis of GitHub repository data from Google Cloud Storage, processing millions of commits, files, and repositories. The project includes data cleaning and quality checks, timeline analysis of commit trends (2008-2022), programming language popularity tracking, license distribution analysis, repository popularity metrics, and technology trend tracking (Docker, Django, Spark, Redis). Advanced analyses include commit message categorization, text similarity analysis using TF-IDF and MinHash LSH, language trends over time, and language-license associations. The project demonstrates expertise in distributed computing, handling large-scale data with appropriate sampling strategies, and performing complex aggregations and joins across multiple parquet datasets.

### UCSB Baseball Internship Work

- **JUCO Player Analysis** - Player evaluation and categorization work from internship with UCSB Baseball, analyzing Junior College (JUCO) hitters and pitchers with school affiliations. The project involves comprehensive player assessment, categorization, and data organization to support recruiting and player evaluation processes. This work demonstrates practical application of baseball analytics in a collegiate setting, with focus on identifying and organizing talent from the JUCO level.

## Writing

- **[What Happened to Bobby Miller?](https://medium.com/@joseph.n.digiovanni/what-happened-to-bobby-miller-40a7e400629f)** - A short case study of the sharp decline in performance from Dodgers' RHP Bobby Miller in 2024, and what he worked on in 2025.

## Technical Skills

- **Programming Languages**: Python, R
- **Machine Learning**: scikit-learn (HistGradientBoostingClassifier, RandomForest), XGBoost, predictive modeling, model deployment, feature engineering
- **Big Data**: PySpark, distributed computing, Google Cloud Storage
- **Data Analysis**: pandas, numpy, statistical modeling (logistic regression, probit models)
- **Visualization**: Plotly, Streamlit dashboards, matplotlib, seaborn
- **Tools**: Jupyter Notebooks, Git, Excel, joblib (model serialization)

---

