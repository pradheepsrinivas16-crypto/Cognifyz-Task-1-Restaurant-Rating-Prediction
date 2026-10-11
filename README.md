# Cognifyz Internship — Task 1: Restaurant Rating Prediction

## Objective
Predict restaurant aggregate ratings using structured restaurant, cuisine, price, service, and location features.

## Notebook
`Task_1_Cognifyz_Restaurant_Rating_Prediction_CV_Safe.ipynb`

The notebook audits the data, excludes likely unrated records (rating 0) from the primary regression task, compares a baseline and four regression models, selects candidates using five-fold cross-validation, tunes Random Forest using training data, and evaluates the selected model on a held-out test set.

## Dataset
The notebook expects the Cognifyz restaurant dataset named `Dataset .csv`. The dataset is not bundled in this ZIP. Upload it when prompted in Colab, or place it beside the notebook for local execution.

## Run in Google Colab
1. Upload the notebook to Colab.
2. Run all cells from top to bottom.
3. Upload `Dataset .csv` when prompted.
4. Use the freshly generated output metrics in any report or README; saved values from previous runs may differ.

## Methods
- Data quality audit and unrated-record handling
- Fold-safe feature engineering in an sklearn pipeline
- One-hot encoding and numeric imputation/scaling
- Mean baseline, Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- Five-fold cross-validation
- GridSearchCV hyperparameter tuning
- Held-out test metrics: R², RMSE, MAE
- Residual diagnostics and permutation importance

## Important limitations
Votes are only available after reviews are submitted. The model with vote count is more appropriate for existing restaurants than brand-new restaurants. Rating text and rating color are excluded because they are derived from the target. Feature importance reflects association, not causation.
