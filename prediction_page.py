import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

# Set a consistent random seed for reproducibility
np.random.seed(42)

# Function to clean data (based on EDA cleaning)
def clean_data(df, target_column=None):
    """
    Cleans the dataset by handling missing values and separating the target column if specified.
    """
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
    
    if num_moderate_cols:
        df_cleaned[num_moderate_cols] = imputer_num.fit_transform(df_cleaned[num_moderate_cols])
    if cat_moderate_cols:
        df_cleaned[cat_moderate_cols] = imputer_cat.fit_transform(df_cleaned[cat_moderate_cols])

    # Impute Missing Values for Columns with Low Missing Values (<5%)
    low_missing_cols = missing_values[(missing_values > 0) & (missing_values <= 0.05 * len(df))].index
    num_low_cols = [col for col in low_missing_cols if df[col].dtype in ['int64', 'float64']]
    cat_low_cols = [col for col in low_missing_cols if df[col].dtype == 'object']
    
    if num_low_cols:
        df_cleaned[num_low_cols] = imputer_num.fit_transform(df_cleaned[num_low_cols])
    if cat_low_cols:
        df_cleaned[cat_low_cols] = imputer_cat.fit_transform(df_cleaned[cat_low_cols])

    # If a target column is provided, separate it from the data
    if target_column and target_column in df_cleaned.columns:
        target_data = df_cleaned[target_column]
        df_cleaned = df_cleaned.drop(columns=[target_column])
        return df_cleaned, target_data
    else:
        return df_cleaned

# Main prediction page function
def prediction_page(df_train, df_test):
    """
    This function sets up the Streamlit app page for house price prediction. It allows users to:
    - Explore the dataset.
    - Select key features and models for prediction.
    - Visualize the results, including actual vs. predicted values and feature importance.
    """
    
    # Explanation for the user
    st.markdown("""
    Welcome to the House Prices Prediction App! This tool allows you to explore the dataset, select features and models, and visualize the prediction results. 
    It is designed to help you understand the factors that influence house prices and make accurate predictions using machine learning models.
    """)

    # Data Exploration Section
    st.sidebar.header("Data Exploration")
    st.sidebar.write("Explore the dataset before making predictions.")
    exploration_choice = st.sidebar.radio("Choose an option:", 
                                          ["None", "View Data", "Visualize Key Features", "Correlation Matrix"])

    # Key features selected based on their relevance to house prices
    key_features = ["OverallQual", "GrLivArea", "YearBuilt", "GarageArea", "GarageYrBlt", "TotRmsAbvGrd"]

    if exploration_choice == "View Data":
        st.write("### Training Data Sample")
        st.write(df_train.head())
        st.write("### Test Data Sample")
        st.write(df_test.head())
        
    elif exploration_choice == "Visualize Key Features":
        st.write("### Key Feature Visualization")
        st.write("Below are visualizations of key features against the SalePrice with a trend line to show their impact on house prices.")
        for feature in key_features:
            fig, ax = plt.subplots()
            sns.regplot(x=df_train[feature], y=df_train['SalePrice'], ax=ax, line_kws={"color": "red"})
            ax.set_title(f'{feature} vs SalePrice with Trend Line')
            st.pyplot(fig)

    elif exploration_choice == "Correlation Matrix":
        st.write("### Correlation Matrix")
        st.write("Visualize the correlations between key features and SalePrice to understand their relationships.")
        key_df = df_train[key_features + ['SalePrice']]
        corr_matrix = key_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Feature Selection for Prediction
    st.sidebar.header("Feature Selection")
    st.sidebar.write("Focus on key features known to predict house prices effectively.")
    selected_features = st.sidebar.multiselect("Select Features:", key_features, default=key_features)

    selected_features = [feature for feature in selected_features if feature in df_test.columns]
    
    # Check if no features are selected
    if not selected_features:
        st.error("No features selected. Please select at least one feature.")
        return

    # Clean and prepare the data
    df_train_cleaned, Y_Train = clean_data(df_train[selected_features + ['SalePrice']], target_column='SalePrice')
    df_test_cleaned = clean_data(df_test[selected_features])

    # Combine training and test data for encoding
    df_combined = pd.concat([df_train_cleaned, df_test_cleaned]).reset_index(drop=True)

    # One-hot encoding for categorical variables (if any)
    object_cols = df_combined.select_dtypes(include='object').columns
    if not object_cols.empty:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = pd.DataFrame(encoder.fit_transform(df_combined[object_cols]))
        encoded_features.index = df_combined.index
        encoded_features.columns = encoder.get_feature_names_out()

        df_final = df_combined.drop(object_cols, axis=1)
        df_final = pd.concat([df_final, encoded_features], axis=1)
    else:
        df_final = df_combined

    # Separate training and test data
    X_Train = df_final.iloc[:len(df_train_cleaned)]
    X_Test = df_final.iloc[len(df_train_cleaned):]

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_Train, Y_Train, train_size=0.8, test_size=0.2, random_state=42)

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    st.sidebar.write("Choose models to evaluate.")
    model_choices = st.sidebar.multiselect("Models:", 
                                           ["Random Forest", "Gradient Boosting", "XGBoost", "Linear Regression", "Decision Tree Regressor", "Ridge", "Lasso"],
                                           default=["Random Forest", "Gradient Boosting", "XGBoost"])

    # Check if no models are selected
    if not model_choices:
        st.error("No models selected. Please select at least one model.")
        return

    # Initialize models
    model_dict = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42)
    }

    # Explain Cross-Validation to the User
    st.write("""
    ### Model Evaluation with Cross-Validation
    Cross-validation is used to evaluate the performance of each model by splitting the training data into multiple subsets (folds).
    The model is trained on some folds and validated on others, ensuring a robust assessment that is less prone to overfitting.
    The average RMSE from cross-validation is used to compare models.
    """)

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    model_performance = {
        'Model': [],
        'MAE': [],
        'RMSE': [],
        'R²': [],
        'Cross-Validated RMSE': [],
        'Feature Importances': [],
    }

    tuned_models = {}
    for model_name in model_choices:
        model = model_dict[model_name]
        tuned_models[model_name] = model

        # Perform cross-validation
        cv_scores = cross_val_score(tuned_models[model_name], X_train, Y_train, cv=kf, scoring='neg_mean_squared_error')
        cv_rmse_scores = np.sqrt(-cv_scores)
        
        # Fit the model on the full training set
        tuned_models[model_name].fit(X_train, Y_train)
        Y_pred = tuned_models[model_name].predict(X_valid)

        # Store performance metrics
        model_performance['Model'].append(model_name)
        model_performance['MAE'].append(mean_absolute_error(Y_valid, Y_pred))
        model_performance['RMSE'].append(np.sqrt(mean_squared_error(Y_valid, Y_pred)))
        model_performance['R²'].append(r2_score(Y_valid, Y_pred))
        model_performance['Cross-Validated RMSE'].append(cv_rmse_scores.mean())

        # Extract feature importances if available
        if hasattr(tuned_models[model_name], 'feature_importances_'):
            feature_importances = tuned_models[model_name].feature_importances_
            sorted_indices = np.argsort(feature_importances)[::-1]

            top_features = []
            for i in sorted_indices[:20]:
                if i < len(df_final.columns):
                    top_features.append(df_final.columns[i])
                else:
                    break

            model_performance['Feature Importances'].append(top_features)
        else:
            model_performance['Feature Importances'].append([])

        # Plot actual vs predicted for the selected model
        st.write(f"### Actual vs Predicted for {model_name}")
        st.write("This plot compares the actual house prices to the predicted prices. The closer the points are to the diagonal line, the better the model's performance.")
        fig, ax = plt.subplots()
        sns.scatterplot(x=Y_valid, y=Y_pred, ax=ax, alpha=0.6)
        ax.set_title(f'Actual vs Predicted for {model_name}')
        ax.set_xlabel('Actual Sale Prices')
        ax.set_ylabel('Predicted Sale Prices')
        ax.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'k--', lw=2)
        st.pyplot(fig)

    # Display model performance
    st.write("### Model Performance Comparison")
    st.write("The table below compares the performance of each selected model using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R²), and Cross-Validated RMSE.")
    performance_df = pd.DataFrame(model_performance)
    st.write(performance_df)

    # Plotting the comparison of MAE, RMSE, and R² across models
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.barplot(x='Model', y='MAE', data=performance_df, ax=ax[0], palette='Blues_d')
    ax[0].set_title('Mean Absolute Error (MAE)')
    ax[0].set_ylabel('MAE')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
    
    sns.barplot(x='Model', y='RMSE', data=performance_df, ax=ax[1], palette='Reds_d')
    ax[1].set_title('Root Mean Squared Error (RMSE)')
    ax[1].set_ylabel('RMSE')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
    
    sns.barplot(x='Model', y='R²', data=performance_df, ax=ax[2], palette='Greens_d')
    ax[2].set_title('R-squared (R²)')
    ax[2].set_ylabel('R²')
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha='right')
    
    st.pyplot(fig)

    # Identify the best model
    best_model_name = performance_df.iloc[performance_df['Cross-Validated RMSE'].idxmin()]['Model']
    best_model_mae = performance_df.iloc[performance_df['Cross-Validated RMSE'].idxmin()]['MAE']
    best_model_rmse = performance_df.iloc[performance_df['Cross-Validated RMSE'].idxmin()]['RMSE']

    st.write(f"### Best Model: `{best_model_name}` with **Cross-Validated RMSE: {best_model_rmse:.2f}** and **MAE: {best_model_mae:.2f}**")
    st.write(f"The `{best_model_name}` model was selected as the best model because it has the lowest Cross-Validated RMSE, indicating the smallest average squared prediction error across all folds.")

    # Hyperparameter tuning for the best model only
    param_grid = {
        "Random Forest": {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "Gradient Boosting": {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "XGBoost": {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        "Ridge": {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
        },
        "Lasso": {
            'alpha': [0.1, 1.0, 10.0],
            'max_iter': [1000, 2000, 5000]
        }
    }

    if best_model_name in param_grid:
        st.write(f"Tuning hyperparameters for the best model: {best_model_name}...")
        search = RandomizedSearchCV(
            tuned_models[best_model_name],
            param_distributions=param_grid[best_model_name],
            n_iter=10,
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, Y_train)
        best_model = search.best_estimator_
        st.write(f"Best parameters for {best_model_name}: {search.best_params_}")

        Y_pred_best = best_model.predict(X_valid)

        # Recalculate metrics with the best hyperparameters
        best_model_mae = mean_absolute_error(Y_valid, Y_pred_best)
        best_model_rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred_best))
        best_model_r2 = r2_score(Y_valid, Y_pred_best)

        st.write(f"### Performance after Hyperparameter Tuning")
        st.write(f"- **RMSE**: {best_model_rmse:.2f}")
        st.write(f"- **MAE**: {best_model_mae:.2f}")
        st.write(f"- **R²**: {best_model_r2:.2f}")
    else:
        best_model = tuned_models[best_model_name]

    # Plot feature importance for the best model
    if hasattr(best_model, 'feature_importances_'):
        st.write("### Feature Importances for the Best Model")
        st.write("The plot below shows the top 6 most important features in the best model. These features have the most influence on predicting house prices.")
        feature_importances = best_model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_n = min(20, len(sorted_indices))  
        top_features = [df_final.columns[i] for i in sorted_indices[:top_n]]
        top_importances = feature_importances[sorted_indices[:top_n]]

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Adjust the left margin to make room for the labels
        plt.subplots_adjust(left=0.3)
        
        sns.barplot(x=top_importances, y=top_features, palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importances for {best_model_name}')
        st.pyplot(fig)

    st.header("Ensemble and Hybrid Model Comparison")

    final_comparison = pd.DataFrame(columns=performance_df.columns)

    best_models = [(model_name, tuned_models[model_name]) for model_name in model_choices]
    ensemble_model = VotingRegressor(estimators=best_models)

    ensemble_model.fit(X_train, Y_train)
    Y_pred_ensemble = ensemble_model.predict(X_valid)

    ensemble_mae = mean_absolute_error(Y_valid, Y_pred_ensemble)
    ensemble_rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred_ensemble))
    ensemble_r2 = r2_score(Y_valid, Y_pred_ensemble)

    ensemble_df = pd.DataFrame({
        'Model': ['Ensemble Model'],
        'MAE': [ensemble_mae],
        'RMSE': [ensemble_rmse],
        'R²': [ensemble_r2],
        'Cross-Validated RMSE': [None],
        'Feature Importances': [None]
    })

    final_comparison = pd.concat([final_comparison, ensemble_df], ignore_index=True)

    if 'Ridge' in tuned_models and 'XGBoost' in tuned_models:
        hybrid_model = VotingRegressor([('ridge', tuned_models['Ridge']), ('xgboost', tuned_models['XGBoost'])])

        hybrid_model.fit(X_train, Y_train)
        Y_pred_hybrid = hybrid_model.predict(X_valid)

        hybrid_mae = mean_absolute_error(Y_valid, Y_pred_hybrid)
        hybrid_rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred_hybrid))
        hybrid_r2 = r2_score(Y_valid, Y_pred_hybrid)

        hybrid_df = pd.DataFrame({
            'Model': ['Hybrid Model'],
            'MAE': [hybrid_mae],
            'RMSE': [hybrid_rmse],
            'R²': [hybrid_r2],
            'Cross-Validated RMSE': [None],
            'Feature Importances': [None]
        })

        final_comparison = pd.concat([final_comparison, hybrid_df], ignore_index=True)

    final_comparison = pd.concat([performance_df, final_comparison], ignore_index=True)

    final_model = None

    if not final_comparison.empty:
        final_model_name = final_comparison.iloc[final_comparison['RMSE'].idxmin()]['Model']
        if final_model_name == 'Ensemble Model':
            final_model = ensemble_model
        elif final_model_name == 'Hybrid Model':
            final_model = hybrid_model
        else:
            final_model = best_model

    st.write("### Final Model Comparison Table")
    st.write(final_comparison)

    if final_model_name:
        st.write(f"#### The final model `{final_model_name}` was selected based on the lowest RMSE and MAE values.")
        st.write(f"- **RMSE**: {final_comparison.iloc[final_comparison['RMSE'].idxmin()]['RMSE']:.2f}")
        st.write(f"- **MAE**: {final_comparison.iloc[final_comparison['MAE'].idxmin()]['MAE']:.2f}")

        st.write("### Final Model Performance Visualization")
        st.write("The bar charts below show the performance of each model in terms of MAE, RMSE, and R².")
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        sns.barplot(x='Model', y='MAE', data=final_comparison, ax=ax[0], palette='Blues_d')
        ax[0].set_title('Mean Absolute Error (MAE)')
        ax[0].set_ylabel('MAE')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')

        sns.barplot(x='Model', y='RMSE', data=final_comparison, ax=ax[1], palette='Reds_d')
        ax[1].set_title('Root Mean Squared Error (RMSE)')
        ax[1].set_ylabel('RMSE')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')

        sns.barplot(x='Model', y='R²', data=final_comparison, ax=ax[2], palette='Greens_d')
        ax[2].set_title('R-squared (R²)')
        ax[2].set_ylabel('R²')
        ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha='right')

        st.pyplot(fig)

        st.write("### Robustness Testing: Subset Training")
        st.write("""
            Robustness testing ensures the model's generalizability by training on different random subsets of the data.
            This helps verify that the model performs consistently across various data splits.
        """)

        random_seeds = [24, 7, 99]
        for seed in random_seeds:
            st.write(f"#### Training with random_state={seed}")
            X_train_sub, X_valid_sub, Y_train_sub, Y_valid_sub = train_test_split(X_Train, Y_Train, train_size=0.8, test_size=0.2, random_state=seed)

            final_model.fit(X_train_sub, Y_train_sub)
            Y_pred_sub = final_model.predict(X_valid_sub)

            mae_sub = mean_absolute_error(Y_valid_sub, Y_pred_sub)
            rmse_sub = np.sqrt(mean_squared_error(Y_valid_sub, Y_pred_sub))
            r2_sub = r2_score(Y_valid_sub, Y_pred_sub)

            st.write(f"- **MAE**: {mae_sub:.2f}")
            st.write(f"- **RMSE**: {rmse_sub:.2f}")
            st.write(f"- **R²**: {r2_sub:.2f}")

    if final_model:
        Y_Pred_test = final_model.predict(X_Test)
        sub = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': Y_Pred_test})
        st.write(f"### Sample of the submission file from the {final_model_name}:")
        st.write(sub.head())

        st.sidebar.download_button(
            label="Download Submission",
            data=sub.to_csv(index=False),
            file_name='submission.csv',
            mime='text/csv'
        )
