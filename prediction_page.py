# Importing all the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

# Set a consistent random seed for reproducibility
np.random.seed(42)

# Function to clean data (based on EDA cleaning)
def clean_data(df):
    # Drop columns with >80% missing values
    missing_values = df.isnull().sum()
    high_missing_cols = missing_values[missing_values > 0.8 * len(df)].index
    df_cleaned = df.drop(columns=high_missing_cols)

    # Impute Missing Values for Columns with Moderate Missing Values (5-80%)
    moderate_missing_cols = missing_values[(missing_values > 0.05 * len(df)) & (missing_values <= 0.8 * len(df))].index
    num_moderate_cols = [col for col in moderate_missing_cols if df[col].dtype in ['int64', 'float64']]
    cat_moderate_cols = [col for col in moderate_missing_cols if df[col].dtype == 'object']
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cleaned[num_moderate_cols] = imputer_num.fit_transform(df_cleaned[num_moderate_cols])
    df_cleaned[cat_moderate_cols] = imputer_cat.fit_transform(df_cleaned[cat_moderate_cols])

    # Impute Missing Values for Columns with Low Missing Values (<5%)
    low_missing_cols = missing_values[(missing_values > 0) & (missing_values <= 0.05 * len(df))].index
    num_low_cols = [col for col in low_missing_cols if df[col].dtype in ['int64', 'float64']]
    cat_low_cols = [col for col in low_missing_cols if df[col].dtype == 'object']
    df_cleaned[num_low_cols] = imputer_num.fit_transform(df_cleaned[num_low_cols])
    df_cleaned[cat_low_cols] = imputer_cat.fit_transform(df_cleaned[cat_low_cols])

    return df_cleaned

# Main prediction page function
def prediction_page(df_train, df_test):
    st.title("House Prices Prediction App")

    # Clean the training and test data
    df_train_cleaned = clean_data(df_train)
    df_test_cleaned = clean_data(df_test)

    # Extract target variable and remove it from training data
    Y_Train = df_train_cleaned['SalePrice']
    df_train_cleaned = df_train_cleaned.drop(columns=['SalePrice'])

    st.write("### Training Data after Cleaning:")
    st.write("Below is a preview of the cleaned training data.")
    st.write(df_train_cleaned.head())

    st.write("### Test Data after Cleaning:")
    st.write("Below is a preview of the cleaned test data.")
    st.write(df_test_cleaned.head())

    # Combine training and test data for encoding
    df_combined = pd.concat([df_train_cleaned, df_test_cleaned]).reset_index(drop=True)

    # One-hot encoding for categorical variables
    object_cols = df_combined.select_dtypes(include='object').columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = pd.DataFrame(encoder.fit_transform(df_combined[object_cols]))
    encoded_features.index = df_combined.index
    encoded_features.columns = encoder.get_feature_names_out()

    # Combine encoded features with the original data
    df_final = df_combined.drop(object_cols, axis=1)
    df_final = pd.concat([df_final, encoded_features], axis=1)

    # Separate training and test data
    X_Train = df_final.iloc[:len(df_train_cleaned)]
    X_Test = df_final.iloc[len(df_train_cleaned):]

    # Split the training data into training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_Train, Y_Train, train_size=0.8, test_size=0.2, random_state=42)

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Gradient Boosting", "XGBoost", "Linear Regression", "Decision Tree Regressor", "Ridge", "Lasso"])

    # Initialize the chosen model
    if model_choice == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_choice == "XGBoost":
        model = XGBRegressor(learning_rate=0.03, n_estimators=200, objective='reg:squarederror', random_state=42)
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=42)
    elif model_choice == "Ridge":
        model = Ridge(random_state=42)
    elif model_choice == "Lasso":
        model = Lasso(random_state=42)

    # Cross-Validation
    st.write("### Cross-Validation Results")
    st.write("""
Cross-validation is a powerful tool for assessing how well your model will generalize to an independent dataset. 
Here, we use k-fold cross-validation (k=5), which means the data is split into 5 subsets. The model is trained 
and validated 5 times, each time using a different subset for validation and the remaining subsets for training.
""")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores)
    st.write(f"Cross-Validation RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")

    # Train the model and make predictions on the validation set
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_valid)

    # Calculate evaluation metrics
    mae = mean_absolute_error(Y_valid, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred))
    r2 = r2_score(Y_valid, Y_pred)

    # Display scatter plot for actual vs. predicted sale prices
    st.write(f"### {model_choice} Model: Actual vs Predicted Sale Prices")
    st.write("""
The scatter plot below shows the relationship between the actual sale prices and the predicted sale prices 
by the selected model. Ideally, the points should align closely with the diagonal line, which represents 
perfect predictions. Deviations from this line indicate prediction errors, with larger deviations suggesting 
larger errors. Analyzing this plot helps in understanding the model's accuracy visually.
""")
    fig, ax = plt.subplots()
    sns.scatterplot(x=Y_valid, y=Y_pred, ax=ax, alpha=0.6)
    ax.set_title(f'{model_choice} Model: Actual vs Predicted Sale Prices')
    ax.set_xlabel('Actual Sale Prices')
    ax.set_ylabel('Predicted Sale Prices')
    ax.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'k--', lw=2)
    st.pyplot(fig)

    # Display metrics for the chosen model
    st.write(f"### Performance Metrics for {model_choice}:")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

    # Subset Training for Robustness Testing
    st.write("### Robustness Testing: Training on Different Subsets")
    st.write("""
To ensure the robustness of our model, we perform training on different subsets of the data using different random 
seeds. This helps us evaluate how the model performs on various splits of the data, revealing potential overfitting 
or underfitting.
""")
    random_seeds = [24, 7, 99]
    for seed in random_seeds:
        X_train_sub, X_valid_sub, Y_train_sub, Y_valid_sub = train_test_split(X_Train, Y_Train, train_size=0.8, test_size=0.2, random_state=seed)
        model.fit(X_train_sub, Y_train_sub)
        Y_pred_sub = model.predict(X_valid_sub)
        rmse_sub = np.sqrt(mean_squared_error(Y_valid_sub, Y_pred_sub))
        r2_sub = r2_score(Y_valid_sub, Y_pred_sub)
        st.write(f"Subset Training RMSE (random_state={seed}): {rmse_sub:.2f}")
        st.write(f"Subset Training R-squared (R²) (random_state={seed}): {r2_sub:.2f}")

    # Compare all models to identify the best one
    st.header("Model Comparison")
    st.write("""
This section compares the performance of different models using three metrics:
- **Mean Absolute Error (MAE):** The average magnitude of errors in a set of predictions, without considering their direction.
- **Root Mean Squared Error (RMSE):** Measures the square root of the average of squared errors, giving more weight to larger errors.
- **R-squared (R²):** Indicates the proportion of variance in the dependent variable that is predictable from the independent variables. A higher R² indicates a better fit.
""")
    models = ["Random Forest", "Gradient Boosting", "XGBoost", "Linear Regression", "Decision Tree Regressor", "Ridge", "Lasso"]
    mae_values = []
    rmse_values = []
    r2_values = []

    # Evaluate each model and store the metrics
    for model_name in models:
        if model_name == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)
        elif model_name == "XGBoost":
            model = XGBRegressor(learning_rate=0.03, n_estimators=200, objective='reg:squarederror', random_state=42)
        elif model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        elif model_name == "Ridge":
            model = Ridge(random_state=42)
        elif model_name == "Lasso":
            model = Lasso(random_state=42)

        # Train and evaluate the model
        model.fit(X_train, Y_train)
        Y_pred_temp = model.predict(X_valid)

        # Calculate metrics
        mae_values.append(mean_absolute_error(Y_valid, Y_pred_temp))
        rmse_values.append(np.sqrt(mean_squared_error(Y_valid, Y_pred_temp)))
        r2_values.append(r2_score(Y_valid, Y_pred_temp))

    # Create a DataFrame to display comparison
    comparison_df = pd.DataFrame({
        'Model': models,
        'MAE': mae_values,
        'RMSE': rmse_values,
        'R²': r2_values
    })

    st.write("### Model Performance Comparison Table")
    st.write(comparison_df)

    # Bar chart for Mean Absolute Error
    st.write("### Mean Absolute Error (MAE) Comparison")
    st.write("""
The following bar chart shows the Mean Absolute Error for each model. Lower values indicate better performance.
""")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='MAE', y='Model', data=comparison_df, palette='Blues_d', ax=ax)
    ax.set_title("Mean Absolute Error (MAE) Comparison")
    ax.set_xscale('log')
    plt.grid(axis='x')
    st.pyplot(fig)

    # Bar chart for Root Mean Squared Error
    st.write("### Root Mean Squared Error (RMSE) Comparison")
    st.write("""
This chart shows the Root Mean Squared Error for each model. Like MAE, lower values indicate better performance,
but RMSE penalizes larger errors more heavily.
""")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='RMSE', y='Model', data=comparison_df, palette='Reds_d', ax=ax)
    ax.set_title("Root Mean Squared Error (RMSE) Comparison")
    ax.set_xscale('log')
    plt.grid(axis='x')
    st.pyplot(fig)

    # Bar chart for R-squared
    st.write("### R-squared (R²) Comparison")
    st.write("""
This chart shows the R-squared values for each model. Higher values indicate better performance, as it shows
how well the model explains the variance in the sale prices.
""")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='R²', y='Model', data=comparison_df, palette='Greens_d', ax=ax)
    ax.set_title("R-squared (R²) Comparison")
    ax.set_xscale('log')
    plt.grid(axis='x')
    st.pyplot(fig)

    # Identify the best model based on the lowest RMSE and MAE
    st.write("### Identifying the Best Model Based on RMSE and MAE")

    # Determine the indices of the models with the lowest RMSE and MAE
    best_rmse_index = np.argmin(rmse_values)
    best_mae_index = np.argmin(mae_values)

    # If both indices point to the same model, it's the clear best model
    if best_rmse_index == best_mae_index:
        best_model_name = models[best_rmse_index]
        st.write(f"### The best model based on both RMSE and MAE is {best_model_name}.")
    else:
        # If different models have the lowest RMSE and MAE, you can either choose one based on priority
        # or introduce additional logic to handle the situation.
        
        st.write(f"The model with the lowest RMSE is {models[best_rmse_index]} with RMSE of {rmse_values[best_rmse_index]:.2f}.")
        st.write(f"The model with the lowest MAE is {models[best_mae_index]} with MAE of {mae_values[best_mae_index]:.2f}.")
        
        # For simplicity, let's choose the model with the lowest RMSE as the final best model, but this can be adjusted
        best_model_name = models[best_rmse_index]
        st.write(f"### Based on the importance of RMSE, {best_model_name} is selected as the best model for final predictions.")

    # Feature Importance Handling
    best_model = None

    if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree Regressor"]:
        if best_model_name == "Random Forest":
            best_model = RandomForestRegressor(random_state=42)
        elif best_model_name == "Gradient Boosting":
            best_model = GradientBoostingRegressor(random_state=42)
        elif best_model_name == "XGBoost":
            best_model = XGBRegressor(random_state=42)
        elif best_model_name == "Decision Tree Regressor":
            best_model = DecisionTreeRegressor(random_state=42)

        # Train the best model
        best_model.fit(X_train, Y_train)
        feature_importances = best_model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_features = [df_final.columns[i] for i in sorted_indices]
        sorted_importances = feature_importances[sorted_indices]

        # Plot feature importance
        st.write(f"### Top 20 Feature Importances for {best_model_name}")
        st.write("The bar chart below shows the top 20 features that contributed most to the model's predictions.")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=sorted_importances[:20], y=sorted_features[:20], palette='Oranges_d', ax=ax)
        ax.set_title(f"Top 20 Feature Importances for {best_model_name}")
        st.pyplot(fig)
        st.write("""
### Understanding Feature Importance
The feature importance chart highlights which factors most influence the model's predictions of house prices.

- **OverallQual** stands out as the most critical factor, indicating that the overall quality of the property has a significant impact on its valuation.
- **GrLivArea** (above-ground living area) and **YearBuilt** (the year the house was built) are also important features, reflecting the value placed on living space and the property's age.
- **GarageArea** and **GarageYrBlt** show that the size and age of the garage play meaningful roles, though less so than the overall quality and living area.
- **TotRmsAbvGrd** (total rooms above ground) is the least influential among the top features, but it still contributes to the model's accuracy.

**Interpretation:** 
- The dominance of **OverallQual** suggests that the subjective quality rating of a property greatly influences its market value. 
- Factors related to the size (e.g., **GrLivArea**) and age (e.g., **YearBuilt**) of the house are also important, though to a lesser extent.
- **Garage features** and **room count** play a more moderate role but should not be overlooked in property valuation.

**Recommendations:**
When assessing or predicting property values, prioritize the overall quality rating, with attention also given to the living area size and property age. Secondary factors like the garage and total room count, while less critical, still provide valuable insights into property valuation.
""")

    # Hyperparameter Tuning for the Best Model
    param_grid = {}

    if best_model_name == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
    elif best_model_name == "Gradient Boosting":
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 4]
        }
    elif best_model_name == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5]
        }
    elif best_model_name == "Decision Tree Regressor":
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == "Ridge":
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
    elif best_model_name == "Lasso":
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'max_iter': [1000, 2000, 5000]
        }

    if param_grid:
        st.write(f"### Tuning hyperparameters for the best model: {best_model_name}")
        st.write("The randomized search will fine-tune the model's parameters to achieve the best possible performance.")
        randomized_search = RandomizedSearchCV(best_model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
        randomized_search.fit(X_train, Y_train)
        best_model = randomized_search.best_estimator_
        st.write(f"**Best Parameters:** {randomized_search.best_params_}")

    # Final evaluation on the validation set with the best model
    final_Y_pred = best_model.predict(X_valid)
    final_mae = mean_absolute_error(Y_valid, final_Y_pred)
    final_rmse = np.sqrt(mean_squared_error(Y_valid, final_Y_pred))
    final_r2 = r2_score(Y_valid, final_Y_pred)

    st.write(f"### Final Model Performance after Hyperparameter Tuning:")
    st.write(f"**Mean Absolute Error:** {final_mae:.2f}")
    st.write(f"**Root Mean Squared Error:** {final_rmse:.2f}")
    st.write(f"**R-squared:** {final_r2:.2f}")

    # Hybrid Model (Ridge + XGBoost)
    st.header("Hybrid Model (Ridge + XGBoost)")

    # Ridge and XGBoost Combination
    ridge_model = Ridge(alpha=1.0, random_state=42)
    hybrid_model = VotingRegressor([('ridge', ridge_model), ('xgboost', best_model)])

    # Train the Hybrid Model
    hybrid_model.fit(X_train, Y_train)
    Y_pred_hybrid = hybrid_model.predict(X_valid)

    # Evaluation Metrics for Hybrid Model before tuning
    hybrid_mae = mean_absolute_error(Y_valid, Y_pred_hybrid)
    hybrid_rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred_hybrid))
    hybrid_r2 = r2_score(Y_valid, Y_pred_hybrid)

    st.write(f"### Initial Performance Metrics for Hybrid Model (Ridge + XGBoost):")
    st.write(f"**Mean Absolute Error (MAE):** {hybrid_mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {hybrid_rmse:.2f}")
    st.write(f"**R-squared (R²):** {hybrid_r2:.2f}")

    # Hyperparameter tuning for the Hybrid Model
    param_grid_hybrid = {
        'ridge__alpha': [0.1, 1.0, 10.0],
        'xgboost__n_estimators': [100, 200],
        'xgboost__learning_rate': [0.01, 0.1],
        'xgboost__max_depth': [3, 4]
    }

    st.write(f"### Tuning hyperparameters for the Hybrid Model:")
    randomized_search_hybrid = RandomizedSearchCV(hybrid_model, param_grid_hybrid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
    randomized_search_hybrid.fit(X_train, Y_train)
    hybrid_best_model = randomized_search_hybrid.best_estimator_

    st.write(f"**Best Parameters for Hybrid Model:** {randomized_search_hybrid.best_params_}")

    # Final evaluation on the validation set with the tuned Hybrid Model
    Y_pred_hybrid_final = hybrid_best_model.predict(X_valid)
    hybrid_final_mae = mean_absolute_error(Y_valid, Y_pred_hybrid_final)
    hybrid_final_rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred_hybrid_final))
    hybrid_final_r2 = r2_score(Y_valid, Y_pred_hybrid_final)

    st.write(f"### Final Hybrid Model Performance after Hyperparameter Tuning:")
    st.write(f"**Mean Absolute Error:** {hybrid_final_mae:.2f}")
    st.write(f"**Root Mean Squared Error:** {hybrid_final_rmse:.2f}")
    st.write(f"**R-squared:** {hybrid_final_r2:.2f}")

    # Ensemble Model (Averaging XGBoost and Hybrid Model)
    st.header("Ensemble Model: Averaging XGBoost and Hybrid Model")
    st.write("""
To further enhance the robustness and performance of our predictions, we can create an ensemble model by averaging 
the predictions from the XGBoost model and the Hybrid Model. This approach leverages the strengths of both models 
to produce more reliable predictions.
""")
    ensemble_predictions = (final_Y_pred + Y_pred_hybrid_final) / 2

    # Evaluation Metrics for Ensemble Model
    ensemble_mae = mean_absolute_error(Y_valid, ensemble_predictions)
    ensemble_rmse = np.sqrt(mean_squared_error(Y_valid, ensemble_predictions))
    ensemble_r2 = r2_score(Y_valid, ensemble_predictions)

    st.write(f"### Ensemble Model Performance:")
    st.write(f"**Mean Absolute Error (MAE):** {ensemble_mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {ensemble_rmse:.2f}")
    st.write(f"**R-squared (R²):** {ensemble_r2:.2f}")

    # Comparison between the best-tuned model, Hybrid Model, and Ensemble Model
    st.header(f"Model Comparison: {best_model_name} vs. Hybrid Model vs. Ensemble Model")
    
    # Create comparison DataFrame
    comparison_results = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R²)'],
        best_model_name: [final_mae, final_rmse, final_r2],
        'Hybrid Model': [hybrid_final_mae, hybrid_final_rmse, hybrid_final_r2],
        'Ensemble Model': [ensemble_mae, ensemble_rmse, ensemble_r2]
    })

    st.write(comparison_results)
    
    # Extract the maximum value across the models for setting the y-axis limits
    max_value = comparison_results.drop('Metric', axis=1).values.max()

    # Visualize Comparison in a Bar Plot
    st.write("### Performance Metrics Comparison")
    fig, ax = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)

    # Best-tuned model Performance
    sns.barplot(x='Metric', y=best_model_name, data=comparison_results, ax=ax[0], color='blue', ci=None)
    ax[0].set_title(f'{best_model_name} Performance')
    ax[0].set_ylim(0, max_value + 5000)
    ax[0].bar_label(ax[0].containers[0])

    # Hybrid Model Performance
    sns.barplot(x='Metric', y='Hybrid Model', data=comparison_results, ax=ax[1], color='green', ci=None)
    ax[1].set_title('Hybrid Model Performance')
    ax[1].set_ylim(0, max_value + 5000)
    ax[1].bar_label(ax[1].containers[0])

    # Ensemble Model Performance
    sns.barplot(x='Metric', y='Ensemble Model', data=comparison_results, ax=ax[2], color='orange', ci=None)
    ax[2].set_title(f'Ensemble Model Performance')
    ax[2].set_ylim(0, max_value + 5000)
    ax[2].bar_label(ax[2].containers[0])

    # Adjusting the plots to improve readability
    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=30, ha='right')
        a.set_ylabel('Score')

    # Show the plot
    st.pyplot(fig)

    # Determine the best model based on the lowest RMSE and MAE
    if (ensemble_rmse < final_rmse) and (ensemble_rmse < hybrid_final_rmse):
        st.write(f"### The Ensemble Model is the best model with the lowest RMSE of {ensemble_rmse:.2f} and MAE of {ensemble_mae:.2f}.")
        Y_Pred_test = (best_model.predict(X_Test) + hybrid_best_model.predict(X_Test)) / 2
        final_model_name = "Ensemble Model"
    elif (hybrid_final_rmse < final_rmse):
        st.write(f"### The Hybrid Model is the best model with the lowest RMSE of {hybrid_final_rmse:.2f} and MAE of {hybrid_final_mae:.2f}.")
        Y_Pred_test = hybrid_best_model.predict(X_Test)
        final_model_name = "Hybrid Model"
    else:
        st.write(f"### The {best_model_name} is the best model with the lowest RMSE of {final_rmse:.2f} and MAE of {final_mae:.2f}.")
        Y_Pred_test = best_model.predict(X_Test)
        final_model_name = best_model_name

    # Prepare the submission file
    sub = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': Y_Pred_test})
    st.write(f"### Sample of the submission file from the {final_model_name}:")
    st.write(sub.head())

    st.sidebar.download_button(
        label="Download Submission",
        data=sub.to_csv(index=False),
        file_name='submission.csv',
        mime='text/csv'
    )

