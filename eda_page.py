import math as m
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats  # Importing stats from scipy for z-score calculation

# Function to remove outliers based on standard deviation and IQR
def remove_outliers(data, column, iqr_multiplier=1.5, z_thresh=3):
    """Remove outliers using IQR and Z-score methods."""
    # IQR method
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(filtered_data[column]))
    filtered_data = filtered_data[(z_scores < z_thresh)]
    
    return filtered_data

# Data Cleaning Function
def clean_data(df):

    st.write("## Data Cleaning: Handling Missing Values and Outliers")
    
    # Display the first 5 rows of the dataframe
    st.write("Displaying the first 5 rows of the dataset:")
    st.dataframe(df.head())
    
    # Display the data types of each column
    st.write("Data types of each column in the dataset:")
    st.write(df.dtypes)
    
    # Display the descriptive statistics of numerical variables
    st.write("Descriptive Statistics of Numerical Variables:")
    st.dataframe(df.describe())
    
    # Display the value counts of the 'SalePrice' column
    st.write("Value counts of the 'SalePrice' column:")
    st.write(df['SalePrice'].value_counts())

    # Display initial missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Visualizing missing values
    st.write("### Missing Values Before Cleaning")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis', ax=ax)
    ax.set_title("Missing Values by Column (Before Cleaning)")
    ax.set_ylabel("Number of Missing Values")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

    # Drop columns with >80% missing values
    high_missing_cols = missing_values[missing_values > 0.8 * len(df)].index
    df_cleaned = df.drop(columns=high_missing_cols)
    st.write(f"Columns dropped due to >80% missing values: {list(high_missing_cols)}")

    # Display table of dropped columns
    if len(high_missing_cols) > 0:
        st.write("### Dropped Columns")
        dropped_columns_df = pd.DataFrame({'Column Name': high_missing_cols})
        st.dataframe(dropped_columns_df)
    else:
        st.write("No columns dropped due to high missing values.")

    # Impute Missing Values for Columns with Moderate Missing Values (5-80%)
    moderate_missing_cols = missing_values[(missing_values > 0.05 * len(df)) & (missing_values <= 0.8 * len(df))].index
    num_moderate_cols = [col for col in moderate_missing_cols if df[col].dtype in ['int64', 'float64']]
    cat_moderate_cols = [col for col in moderate_missing_cols if df[col].dtype == 'object']
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cleaned[num_moderate_cols] = imputer_num.fit_transform(df_cleaned[num_moderate_cols])
    df_cleaned[cat_moderate_cols] = imputer_cat.fit_transform(df_cleaned[cat_moderate_cols])
    st.write(f"Columns imputed (moderate missing values): {list(moderate_missing_cols)}")

    # Impute Missing Values for Columns with Low Missing Values (<5%)
    low_missing_cols = missing_values[(missing_values > 0) & (missing_values <= 0.05 * len(df))].index
    num_low_cols = [col for col in low_missing_cols if df[col].dtype in ['int64', 'float64']]
    cat_low_cols = [col for col in low_missing_cols if df[col].dtype == 'object']
    df_cleaned[num_low_cols] = imputer_num.fit_transform(df_cleaned[num_low_cols])
    df_cleaned[cat_low_cols] = imputer_cat.fit_transform(df_cleaned[cat_low_cols])
    st.write(f"Columns imputed (low missing values): {list(low_missing_cols)}")

    # Verify that there are no missing values left
    missing_values_after = df_cleaned.isnull().sum().sum()

    st.write(f"Total missing values after cleaning: {missing_values_after}")

    return df_cleaned

def eda_page(df_train, df_test):
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Understand the data through statistical summaries, visualizations, and key insights to guide house price predictions.")

    # Data Cleaning
    df_train_cleaned = clean_data(df_train)

    st.write("### Training Data Overview:")
    st.write(df_train_cleaned.describe())
    st.write("Summary statistics provide an overview of the training data, including central tendency and spread of the features.")

    st.write("### Test Data Overview:")
    st.write(df_test.describe())
    st.write("Summary statistics for the test data help in understanding how it compares to the training data.")

    # Visualizations
    st.title("Visualizations")
    
    # Sale Price Distribution
    st.write("### Sale Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_train['SalePrice'], kde=True, bins=30, ax=ax)
    ax.set_xlabel('Sale price')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write("""
    The histogram provides a visual representation of house sale prices, illustrating the frequency and distribution of different price intervals.
    It highlights common price ranges and potential outliers, offering insights into market trends and pricing strategies.
    """)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    numeric_df = df_train_cleaned.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(20, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    st.write("""
    The correlation heatmap shows relationships between features, helping to identify key predictors for house prices.
    Strong correlations are highlighted, revealing important relationships like the positive correlation between OverallQual (quality) and SalePrice.
    """)
    st.write("### Correlation Analysis Summary")
    st.write("""
    The correlation analysis reveals several important relationships:
    - **OverallQual and SalePrice:** Strong positive correlation (0.79), indicating higher quality homes are priced higher.
    - **GrLivArea and SalePrice:** Positive correlation (0.71), showing that larger living areas increase house prices.
    - **GarageCars and SalePrice:** Correlation (0.64) suggests that garage capacity contributes to house value.
    - **TotalBsmtSF and SalePrice:** Correlation (0.61) indicates that larger basements are valued higher.
    - **YearBuilt and SalePrice:** Correlation (0.52) reflects the impact of newer constructions on price.
    """)

    
    # Distribution of Selected Numerical Features
    st.write("### Distribution of Selected Numerical Features")
    numerical_features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea', 'TotalBsmtSF']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, feature in enumerate(numerical_features):
        sns.histplot(df_train_cleaned[feature], bins=30, kde=True, ax=axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(f'Distribution of {feature}')
        axes[i // 3, i % 3].set_xlabel(feature)
        axes[i // 3, i % 3].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("""
    These histograms display the distribution of selected numerical features, revealing important patterns:
    - **LotArea:** Right-skewed, with some large outliers.
    - **OverallQual:** Nearly normal, centered around average quality ratings.
    - **OverallCond:** Slightly right-skewed, centered around average condition ratings.
    - **YearBuilt:** Peaks around recent years, showing trends in newer constructions.
    - **GrLivArea:** Right-skewed, indicating most houses have moderate living areas, with some outliers.
    - **TotalBsmtSF:** Similar to GrLivArea, with right-skewness and outliers indicating larger basement areas.
    """)

    # Scatter Plots of Key Features vs. Sale Price
    st.write("### Scatter Plots of Key Features vs. Sale Price")
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']
    for i, feature in enumerate(features):
        sns.scatterplot(x=feature, y='SalePrice', data=df_train_cleaned, ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(f'{feature} vs. SalePrice')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("""
    These scatter plots illustrate the relationships between key features and house sale prices:
    - **OverallQual vs. SalePrice:** Higher quality ratings correlate with higher prices.
    - **GrLivArea vs. SalePrice:** Larger living areas generally lead to higher prices.
    - **GarageCars vs. SalePrice:** The number of garage spaces shows a positive correlation with sale price.
    - **GarageArea vs. SalePrice:** The garage area is positively related to sale price, similar to garage capacity.
    - **TotalBsmtSF vs. SalePrice:** Larger basements correspond with higher sale prices.
    - **YearBuilt vs. SalePrice:** Newer homes tend to have higher sale prices, reflecting their modern features and appeal.
    """)

    # Outlier Removal
    st.title("Outlier Removal")
    st.write("### Boxplot with Outliers")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(df_train_cleaned["SalePrice"], ax=ax)
    ax.set_title("Boxplot of SalePrice with Outliers")
    st.pyplot(fig)
    st.write("""
    The boxplot reveals the presence of outliers in the SalePrice variable. Outliers can skew the analysis and impact the accuracy of predictive models, so it is crucial to address them.
    """)

    # Outlier removal process using combined IQR and Z-score method
    st.write("### Outlier Removal Process")
    df_train_no_outliers = remove_outliers(df_train_cleaned, 'SalePrice')
    st.write("Removed outliers from 'SalePrice' using a combined IQR and Z-score method.")

    # Reset the index
    df_train_no_outliers.reset_index(drop=True, inplace=True)
    st.write("Data after outlier removal:")
    st.dataframe(df_train_no_outliers.describe())

    # Boxplot after outlier removal
    st.write("### Boxplot After Outlier Removal")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(df_train_no_outliers["SalePrice"], ax=ax)
    ax.set_title("Boxplot of SalePrice After Outlier Removal")
    st.pyplot(fig)
    st.write("""
    After removing outliers, the boxplot shows a more accurate representation of the SalePrice distribution, with extreme values eliminated. This step helps in achieving more reliable statistical analyses and predictions.
    """)

    # Additional Visualizations: Feature vs. SalePrice for Categorical Data
    st.write("### Sales Price by Neighborhood")
    price_by_neighborhood = df_train_no_outliers.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
    df_price_by_neighborhood = pd.DataFrame({'Neighborhood': price_by_neighborhood.index, 'PriceMean': price_by_neighborhood.values})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_price_by_neighborhood, x='Neighborhood', y='PriceMean', ci=None, palette='Set2', ax=ax)
    ax.set_title('Average Sales Price by Neighborhood')
    ax.set_xlabel('Neighborhood')
    ax.set_ylabel('Average Selling Price')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)
    st.write("""
    This bar chart shows the average sales prices by neighborhood. 'NoRidge' stands out as the highest-priced neighborhood, likely due to its desirable location and high-quality housing. This visualization helps identify premium areas within the dataset.
    """)

    # Additional Visualizations: Categorical Variable Analysis
    st.write("### Distribution of Overall Quality Ratings")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_train_no_outliers, x='OverallQual', palette='Set2', order=df_train_no_outliers['OverallQual'].value_counts().index, ax=ax)
    ax.set_title('General Quality Rating Distribution')
    ax.set_xlabel('Overall Quality Rating')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write("""
    The bar chart displays the distribution of overall quality ratings. Most houses are rated between 5 and 7, indicating a concentration of moderate to high-quality homes in the dataset.
    """)

    # Additional Analysis: Relationship between House Style and SalePrice
    st.write("### Sales Price by House Style")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_train_no_outliers, x='HouseStyle', y='SalePrice', palette='Set2', order=df_train_no_outliers.groupby('HouseStyle')['SalePrice'].median().sort_values(ascending=False).index, ax=ax)
    ax.set_title('Sales Price by Type of Construction')
    ax.set_xlabel('Construction Type')
    ax.set_ylabel('Sale price')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("""
    This boxplot shows the sales price by house style. It highlights the variability in prices based on different architectural styles, with certain styles commanding higher prices than others.
    """)

    # Additional Visualizations: Total Built Area vs. SalePrice
    st.write("### Relationship between Total Built Area and Sale Price")
    df_train_no_outliers['TotalSF'] = df_train_no_outliers['TotalBsmtSF'] + df_train_no_outliers['1stFlrSF'] + df_train_no_outliers['2ndFlrSF']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_train_no_outliers, x='TotalSF', y='SalePrice', palette='Set2', alpha=0.6, ax=ax)
    ax.set_title('Relationship between Total Built Area and Sales Price')
    ax.set_xlabel('Total Built Area (sqft)')
    ax.set_ylabel('Selling Price')
    st.pyplot(fig)
    st.write("""
    The scatter plot shows a positive correlation between the total built area and sales price. Larger homes tend to have higher sale prices, reflecting the value of space in real estate.
    """)

    # Additional Visualizations: Year Built vs. SalePrice
    st.write("### Sales Price Over Time by Year Built")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_train_no_outliers, x='YearBuilt', y='SalePrice', palette='Set2', estimator=np.median, ax=ax)
    ax.set_title('Relationship between Sales Price and Year of Construction')
    ax.set_xlabel('Construction Year')
    ax.set_ylabel('Sales Price (Median)')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("""
    This time series analysis shows the relationship between the year a house was built and its sale price. Newer homes generally command higher prices, reflecting their modern features and desirability.
    """)
    
    # Conclusion
    st.title("Conclusion of Data Analysis")
    st.write("""
    The data analysis of the house price dataset reveals several critical factors influencing property values:

    1. **Overall Quality (OverallQual):** This is the most significant predictor of house prices. Higher quality construction and finishing result in higher sales prices.
    
    2. **Living Area (GrLivArea):** Larger homes with more above-ground living space tend to fetch higher prices, making this another crucial factor.
    
    3. **Garage Capacity and Area:** Both the number of cars a garage can hold and its size are important in determining property value.
    
    4. **Year Built:** Newer homes are generally more expensive, reflecting their modern amenities and construction quality.
    
    5. **Basement Area (TotalBsmtSF):** Larger basements contribute positively to house prices, especially when they are finished.
    
    6. **Neighborhood:** Location plays a key role, with certain neighborhoods commanding significantly higher prices than others.

    **Actionable Insights:**
    - **For Buyers:** Focus on properties with high OverallQual and ample living space, as these are strong indicators of value.
    - **For Sellers:** Enhancing the quality of finishes and adding space, such as extending the living area or improving the garage, can significantly increase property value.
    - **For Investors:** Consider properties in high-demand neighborhoods like 'NoRidge' for potential appreciation and higher returns.

    The removal of outliers and detailed exploration of features ensure that subsequent predictive models will be based on robust and accurate data, leading to more reliable predictions.
    """)



