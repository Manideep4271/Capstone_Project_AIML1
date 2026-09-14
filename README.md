Capstone Project — AI & Machine Learning
Certificate Program in Artificial Intelligence and Machine Learning

This repository contains the complete capstone project implementation covering data collection and SQL processing, exploratory analytics and machine learning, and a policy-based support assistant.

Repository Structure
capstone-ai-ml/
│
├── README.md
│
├── data_pipeline/
│   ├── data/
│   │   └── books_cleaned.csv
│   ├── output/
│   │   └── query_outputs.txt
│   ├── zepto_books.db
│   ├── module1_books_pipeline.ipynb
│   └── README.md
│
├── analytics/
│   ├── titanic.csv
│   │
│   ├── models/
│   │   ├── titanic_classification_pipeline.joblib
│   │   └── titanic_fare_regression_pipeline.joblib
│   │
│   ├── outputs/
│   │   ├── classification_metrics.csv
│   │   ├── regression_metrics.csv
│   │   ├── correlation_matrix.csv
│   │   ├── outlier_summary.csv
│   │   └── imbalance_comparison.csv
│   │
│   ├── module2_titanic_analytics.ipynb
│   └── README.md
│
└── support_assistant/
    ├── data/
    │   └── policies/
    ├── chroma_db/
    ├── app/
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
Module 1 — Data Pipeline
Objective

Module 1 implements an end-to-end data pipeline for scraping book information from Books to Scrape, cleaning the data, converting prices, storing the data in a normalized SQLite database, and executing SQL analysis queries.

The complete process is automated and does not require manual copy-paste of book data.

Technologies
Python
Requests
BeautifulSoup
Pandas
SQLite
SQL
Data Collection

The pipeline collects book information from:

https://books.toscrape.com/

The scraper processes multiple catalogue pages and collects at least 60 books across multiple categories.

The following fields are collected:

Title
Price in GBP
Star rating
Availability
Category
Data Cleaning

The scraped data is converted into the following cleaned fields:

Field	Description
title	Book title
price_gbp	Numeric price in GBP
price_inr	Price converted to INR
rating	Rating from 1 to 5
in_stock	Boolean stock status
category	Book category

The fixed conversion rate required by the assignment is:

1 GBP = 105.50 INR

No external currency API is used.

Database Design

The cleaned data is stored in SQLite using normalized tables.

Categories
categories
-----------
category_id PRIMARY KEY
category_name
Books
books
-----
book_id PRIMARY KEY
title
price_gbp
price_inr
rating
in_stock
category_id FOREIGN KEY

The foreign-key relationship connects each book to its category.

SQL Analysis

The project includes SQL queries demonstrating:

SELECT / WHERE
ORDER BY
LIMIT
DISTINCT
BETWEEN
JOIN

The SQL query strings and outputs are stored in:

data_pipeline/output/query_outputs.txt
Pandas Validation

At least two SQL results are loaded using:

pd.read_sql()

The SQL JOIN result is also reproduced using:

pd.merge()

The two approaches are compared to verify equivalent results.

Module 1 Output
data_pipeline/
├── data/
│   └── books_cleaned.csv
├── output/
│   └── query_outputs.txt
└── zepto_books.db
Module 2 — Analytics and Machine Learning
Objective

Module 2 performs exploratory data analysis and machine learning using the Titanic dataset.

The module covers data cleaning, statistical analysis, visualization, classification, class-imbalance handling, hyperparameter tuning, regression, model evaluation, and model persistence.

Dataset

The Titanic dataset is initially loaded using:

sns.load_dataset("titanic")

It is immediately saved locally as:

analytics/titanic.csv

This provides an offline fallback for subsequent processing.

Exploratory Data Analysis

The following analyses are performed:

Dataset structure
Data types
Missing values
Missing-value percentages
Age distribution
Fare distribution
Age outliers
Fare outliers
Fare mean
Fare median
Fare mode
Fare skewness
Missing-Value Strategy

The project evaluates missing values based on their percentage.

The strategy follows the assignment requirements:

< 5%       → Drop rows
5% – 30%   → Impute
High       → Drop or encode with justification

For machine-learning pipelines, numeric missing values are handled using median imputation and categorical missing values using most-frequent imputation.

Survival Analysis

Survival is analyzed using:

Sex
Passenger class
Sex + passenger class

Visualizations are generated to interpret differences in survival rates.

Correlation Analysis

The correlation matrix contains exactly these six variables:

survived
pclass
age
sibsp
parch
fare

The following fields are intentionally excluded:

adult_male
alone

A correlation heatmap is generated and the two strongest absolute correlations are identified.

Standardization

Numeric features are standardized using z-score standardization.

A sanity check verifies that the standardized variables have approximately:

Mean = 0
Standard deviation = 1
Classification

The target variable is:

survived

The following models are evaluated:

Logistic Regression
Decision Tree
Random Forest

The train/test split is stratified and performed before preprocessing.

Preprocessing

The machine-learning preprocessing is implemented using:

Pipeline
ColumnTransformer
SimpleImputer
StandardScaler
OneHotEncoder

Numeric features are median-imputed and standardized.

Categorical features are most-frequent imputed and one-hot encoded.

This prevents preprocessing information from the test set from leaking into model training.

Classification Metrics

Each classification model is evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC

Confusion matrices and ROC curves are also generated.

Class Imbalance

Three approaches are compared:

Baseline
class_weight="balanced"
SMOTE

SMOTE is applied only to the training data.

Random Forest Hyperparameter Tuning

Random Forest is tuned using GridSearchCV.

The search includes:

n_estimators
max_depth
max_features

The Random Forest model uses:

oob_score=True

The best model is evaluated on the test set.

Fare Regression

A regression model is developed to predict:

fare

The regression model uses passenger information such as:

Passenger class
Sex
Age
Siblings/spouses
Parents/children
Embarked port

The regression model is evaluated using:

MAE
RMSE
R²
Adjusted R²

Residual analysis is performed to investigate possible heteroscedasticity.

Saved Models

The complete preprocessing and estimator pipelines are saved using Joblib:

analytics/models/
├── titanic_classification_pipeline.joblib
└── titanic_fare_regression_pipeline.joblib

The saved pipelines are reloaded and tested to verify that they work correctly after serialization.

Module 2 Outputs
analytics/
├── titanic.csv
│
├── models/
│   ├── titanic_classification_pipeline.joblib
│   └── titanic_fare_regression_pipeline.joblib
│
├── outputs/
│   ├── classification_metrics.csv
│   ├── regression_metrics.csv
│   ├── correlation_matrix.csv
│   ├── outlier_summary.csv
│   └── imbalance_comparison.csv
│
└── module2_titanic_analytics.ipynb
Module 3 — Policy Support Assistant
Objective

Module 3 implements a local policy-based support assistant that retrieves relevant policy information and generates structured answers.

The assistant is designed to work without a network-based LLM when:

MOCK_LLM=1

This is the default grading mode.

Technologies
Python
Sentence Transformers
all-MiniLM-L6-v2
ChromaDB
LangGraph
Pydantic
FastAPI
Docker
Policy Documents

The assistant uses eight local policy documents covering the required support topics.

The documents are stored locally under:

support_assistant/data/policies/

No paid external service is required.

Embeddings

Policy documents are converted into embeddings using the local model:

all-MiniLM-L6-v2

The embeddings are stored in ChromaDB.

This allows policy questions to be matched against the most relevant policy content.

Retrieval

For policy-related questions, the assistant performs semantic retrieval and returns the top three relevant chunks.

Cosine similarity is used to determine relevance.

The retrieved context is then used to produce the answer.

In mock mode, the response follows the required format:

Based on the retrieved context: {top_chunk_snippet}
Supported Intent Categories

The assistant recognizes policy-related intents using keywords including:

delivery
return
refund
membership
tracking
cancel
gift card
support hours

General questions that do not match these policy categories are handled using the direct-answer path.

Prompt Structure

The assistant uses a structured prompt containing:

Role
Context
Task
Format
Length

A negative constraint is also included to prevent unsupported answers.

Few-shot examples are used to demonstrate the expected response format.

LangGraph Architecture

The assistant uses a LangGraph StateGraph with three named nodes:

classify_intent
        |
        v
  policy query?
     /       \
   yes        no
   |           |
   v           v
retrieve_   direct_
and_answer  answer

The three nodes are:

classify_intent
retrieve_and_answer
direct_answer

The graph state is represented using a TypedDict.

Structured Response

Responses are validated using Pydantic.

The response contains:

answer
sources
confidence

This ensures that the API returns a consistent structured response.

FastAPI

The assistant exposes:

POST /ask

The endpoint accepts a user question and returns the structured assistant response.

Example request:

{
  "question": "What is the return policy?"
}
Mock LLM Mode

The application defaults to mock mode:

MOCK_LLM=1

or when the variable is unset.

This allows the complete application to run without requiring an external LLM or network connection.

Docker

The module includes a Dockerfile for containerized execution.

The application can therefore be packaged and executed consistently across environments.

Module 3 Structure
support_assistant/
├── data/
│   └── policies/
│       ├── policy_1.txt
│       ├── policy_2.txt
│       ├── policy_3.txt
│       ├── policy_4.txt
│       ├── policy_5.txt
│       ├── policy_6.txt
│       ├── policy_7.txt
│       └── policy_8.txt
│
├── chroma_db/
│
├── app/
│   ├── main.py
│   ├── graph.py
│   ├── retrieval.py
│   ├── schemas.py
│   └── config.py
│
├── requirements.txt
├── Dockerfile
└── README.md
Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd capstone-ai-ml

Each module contains its own requirements and execution instructions.

Module Execution
Module 1

Navigate to:

cd data_pipeline

Install dependencies:

pip install -r requirements.txt

Run the pipeline according to the instructions in:

data_pipeline/README.md

The pipeline generates the cleaned CSV, SQLite database, and SQL query output.

Module 2

Open:

analytics/module2_titanic_analytics.ipynb

Run the notebook cells sequentially.

The notebook creates:

analytics/titanic.csv
analytics/models/
analytics/outputs/
Module 3

Navigate to:

cd support_assistant

Install dependencies:

pip install -r requirements.txt

Set mock mode:

export MOCK_LLM=1

Run the FastAPI application using the instructions in:

support_assistant/README.md
Reproducibility

The project is designed to be reproducible.

Module 1 automatically collects and processes source data.

Module 2 saves the Titanic dataset locally immediately after loading it.

Module 3 uses local policy documents and local embeddings.

Random seeds are used where applicable for machine-learning experiments.

The main machine-learning random state is:

42
Key Design Decisions
Module 1
Automated scraping rather than manual data entry
BeautifulSoup for HTML parsing
Pandas for data cleaning
Fixed GBP-to-INR conversion rate
Normalized SQLite schema
SQL and Pandas result validation
Module 2
Immediate offline Titanic CSV fallback
Train/test split before preprocessing
Pipeline and ColumnTransformer to prevent leakage
Multiple classification models
Class imbalance comparison
Random Forest hyperparameter tuning
Separate regression analysis
Joblib model persistence
Module 3
Local policy documents
Local sentence-transformer embeddings
ChromaDB semantic retrieval
LangGraph workflow
Pydantic response validation
FastAPI API
Mock LLM mode for offline grading
Docker support
Testing and Validation

The project includes validation for:

Dataset size and required fields
SQL query execution
SQL JOIN versus Pandas merge
Correlation matrix requirements
Classification metrics
Class imbalance approaches
Random Forest OOB score
Regression metrics
Model serialization and reload
Support-assistant retrieval
Structured API responses
Git Workflow

The project should be developed using a feature branch and merged into the main branch.

Example:

git checkout -b feature/capstone-modules

Make commits during development:

git add .
git commit -m "Implement Module 1 data pipeline"
git add .
git commit -m "Implement Module 2 analytics and ML"
git add .
git commit -m "Implement Module 3 support assistant"

Push the branch:

git push origin feature/capstone-modules

Create a pull request and merge the feature branch into main.

Academic Integrity

This project is intended as an academic capstone submission.

All implementation should be reviewed and understood by the student before submission.

No paid services are required for the project.

The repository contains source code, data outputs, configuration, and documentation required to reproduce the work.

Final Submission Checklist
Module 1

Automated scraping implemented

At least 60 books collected

At least 3 categories covered

Required fields present

Price converted to numeric GBP

Rating converted to integer 1–5

Stock converted to boolean

GBP → INR conversion uses 105.50

SQLite normalized database created

At least 5 SQL queries included

SQL JOIN included

Query outputs saved

pd.read_sql() demonstrated

pd.merge() JOIN equivalence demonstrated

Module 2

Titanic loaded using sns.load_dataset('titanic')

analytics/titanic.csv created immediately

Missing-value strategy documented

Age histogram and boxplot created

Fare histogram and boxplot created

IQR outlier analysis completed

Fare mean/median/mode calculated

Fare skew analyzed

Survival by sex analyzed

Survival by passenger class analyzed

Survival by sex + passenger class analyzed

Exactly six-column correlation matrix created

Heatmap created

Strongest two absolute correlations identified

At least four charts interpreted

Z-score sanity check completed

Stratified train/test split completed

Pipeline + ColumnTransformer implemented

Logistic Regression evaluated

Decision Tree evaluated

Random Forest evaluated

Baseline/class-weight/SMOTE compared

GridSearchCV completed

OOB score reported

Confusion matrix generated

Accuracy/precision/recall/F1/ROC-AUC reported

Fare regression completed

MAE/RMSE/R²/Adjusted R² reported

Residual analysis completed

Joblib pipeline saved

Saved pipeline reloaded and tested

Module 3

Eight policy documents included

Local all-MiniLM-L6-v2 embeddings used

ChromaDB used

Structured prompt implemented

Negative constraint included

Few-shot examples included

LangGraph StateGraph implemented

TypedDict state implemented

classify_intent node implemented

retrieve_and_answer node implemented

direct_answer node implemented

Top-3 policy retrieval implemented

Cosine similarity used

Mock LLM mode supported

Pydantic response model implemented

FastAPI POST /ask implemented

Dockerfile included

Module README included

