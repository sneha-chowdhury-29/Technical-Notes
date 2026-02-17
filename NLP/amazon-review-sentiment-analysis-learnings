# Technical Learnings from Amazon Review Sentiment Analysis Project

## Problem Framing

This project focused on building a binary sentiment classifier for Amazon product reviews. The objective was to:

- Convert raw textual reviews into structured features
- Train a classification model
- Handle class imbalance
- Evaluate performance rigorously
- Deploy the model via Streamlit

The task was framed as a supervised binary classification problem.

---

## Data Engineering & Label Design

### Rating to Label Conversion

The original dataset contained ratings in textual form (e.g., "Rated 4 out of 5 stars").

Key learning:
- Real-world datasets often require label engineering before modeling.
- Regex-based extraction was used to convert ratings to numeric values.
- Binary labels were created:
  - `1` → Positive (rating >= 4)
  - `0` → Negative (rating <= 2)
  - Neutral reviews were excluded to simplify the classification boundary.

This highlighted the importance of defining business-aligned label logic.

---

## Text Representation: Feature Engineering

### TF-IDF Vectorization

Text data was converted into numerical features using:

- `TfidfVectorizer`
- Unigrams + Bigrams (`ngram_range=(1,2)`)
- Limited vocabulary size (`max_features=5000`)

Key learnings:

- TF-IDF remains a strong baseline for classical NLP tasks.
- Increasing features does not always improve performance.
- Removing stopwords can sometimes degrade sentiment performance (e.g., removing "not").
- Bigrams improve sentiment capture (e.g., "not good", "very bad").

This reinforced that feature design has a direct impact on model performance.

---

## Model Comparison & Assumptions

### Naive Bayes vs Logistic Regression

Two classical models were compared:

- Multinomial Naive Bayes
- Logistic Regression

Key insights:

- Naive Bayes assumes conditional independence of features.
- Logistic Regression learns weighted feature contributions.
- Logistic Regression performed better in minority class F1-score.
- Linear models work exceptionally well with TF-IDF features.

This demonstrated the importance of understanding model assumptions, not just performance numbers.

---

## Handling Class Imbalance

The dataset was imbalanced (~70% negative, ~30% positive).

Initial model bias favored the majority class.

Solution implemented:
- `class_weight='balanced'` in Logistic Regression

Impact:
- Increased minority recall
- Slight decrease in precision
- Improved minority F1-score

Key learning:
- Accuracy alone is misleading in imbalanced datasets.
- Precision-recall trade-offs must be evaluated explicitly.

---

## Evaluation Strategy

### Metrics Used

- Precision
- Recall
- F1-score
- Confusion Matrix
- Stratified Cross-Validation

### Important Learning

Using simple `cross_val_score` caused failures due to imbalance.

Solution:
- Used `StratifiedKFold` to preserve class distribution across folds.

This reinforced:

- Always use stratified splitting for classification problems.
- Cross-validation ensures stability beyond a single train-test split.

---

## Pipeline Engineering

Manual vectorization and modeling were replaced with a `sklearn.pipeline.Pipeline`.

Benefits:

- Prevented data leakage
- Ensured reproducibility
- Simplified inference
- Made deployment cleaner

This emphasized the importance of engineering discipline over experimentation.

---

## Model Persistence & Deployment

The trained pipeline was saved using `joblib`.

A Streamlit application was built to:

- Accept user input
- Load the trained model
- Generate real-time predictions

Key learning:
- Deployment constraints differ from notebook experimentation.
- Applications must handle empty input and runtime environments properly.

---

## Practical Debugging Lessons

During development:

- NaN values broke TF-IDF during cross-validation.
- Running Streamlit inside Jupyter caused runtime warnings.
- Increasing feature space did not necessarily improve results.

Key insight:
Robust ML systems require careful data validation and environment awareness.

---

## Overall Takeaways

This project demonstrated:

- End-to-end ML pipeline development
- Feature engineering for NLP
- Model comparison and trade-off analysis
- Handling class imbalance
- Proper evaluation practices
- Deployment considerations
- Reproducibility via pipelines

Most importantly, it reinforced that:
- Clean engineering practices matter more than complex models.
- Classical ML models remain strong baselines in text classification tasks.
