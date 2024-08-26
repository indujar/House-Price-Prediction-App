import streamlit as st
import pandas as pd
from prediction_page import prediction_page
from eda_page import eda_page
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Sidebar for theme selection
theme_choice = st.sidebar.radio("Choose Theme", ["Dark", "Light"])

# Apply custom CSS based on theme choice
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .main {
            background-color: black;
            color: white;
        }
        [data-testid=stSidebar] {
            background-color: #333333;
        }
        [data-testid=stSidebarHeader] {
            background-color: white;
            height: 3.75rem;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: white;
        }
        .stTextArea label {
            color: white;
        }
        .stButton button label {
            color: black;
        }
        .stFileUploaderFileData {
            color: white;
        }
        .stFileUploaderDeleteBtn  {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .main {
            background-color: white;
            color: black;
        }
        [data-testid=stSidebar] {
            background-color: #f0f0f0;
        }
        [data-testid=stSidebarHeader] {
            background-color: black;
            height: 3.75rem;
            color: black;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: black;
        }
        .stTextArea label {
            color: black;
        }
        .stButton button label {
            color: white;
        }
        .stFileUploaderFileData {
            color: black;
        }
        .stFileUploaderDeleteBtn  {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["About", "Prediction", "EDA"])

# Upload files
st.sidebar.header("Upload Data")
st.sidebar.write("Upload training and test data to predict house prices")
train_file = st.sidebar.file_uploader("Upload Training Data", type=["csv"], help="Upload the CSV file that contains your training data.")
test_file = st.sidebar.file_uploader("Upload Test Data", type=["csv"], help="Upload the CSV file that contains your test data.")

# Check if files are uploaded
if train_file is not None and test_file is not None:
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    st.session_state["df_train"] = df_train
    st.session_state["df_test"] = df_test
    data_loaded = True
else:
    df_train = df_test = None
    data_loaded = False

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["About", "Prediction", "EDA"])

# About page content
with tab1:
    st.title("About This App")
    st.write("""
    This app is designed to predict house prices based on various features such as size, location, and quality.
    It is intended for real estate professionals, data analysts, and anyone interested in understanding 
    the factors that influence house prices.
    """)
    st.subheader("How It Works")
    st.write("""
    The app uses a dataset with features that describe houses, such as their square footage, 
    the number of rooms, and the year they were built. A machine learning model is trained on this 
    data to predict house prices for unseen properties.
    """)
    st.subheader("Key Data Fields")
    st.write("""
    Here are some of the key fields in the dataset:

    - **SalePrice:** The property's sale price in dollars. This is the target variable that you're trying to predict.
    - **MSSubClass:** The building class.
    - **MSZoning:** The general zoning classification.
    - **LotFrontage:** Linear feet of street connected to property.
    - **LotArea:** Lot size in square feet.
    - **OverallQual:** Overall material and finish quality.
    - **YearBuilt:** Original construction date.
    - **GrLivArea:** Above grade (ground) living area square feet.
    - **GarageCars:** Size of garage in car capacity.
    - **TotalBsmtSF:** Total square feet of basement area.
    """)

    if st.checkbox("Show all data fields"):
        st.write("""
        Here is a brief overview of all the variables in the dataset:

        - **MSSubClass:** The building class.
        - **MSZoning:** The general zoning classification.
        - **LotFrontage:** Linear feet of street connected to property.
        - **LotArea:** Lot size in square feet.
        - **Street:** Type of road access.
        - **Alley:** Type of alley access.
        - **LotShape:** General shape of property.
        - **LandContour:** Flatness of the property.
        - **Utilities:** Type of utilities available.
        - **LotConfig:** Lot configuration.
        - **LandSlope:** Slope of property.
        - **Neighborhood:** Physical locations within Ames city limits.
        - **Condition1:** Proximity to main road or railroad.
        - **Condition2:** Proximity to main road or railroad (if a second is present).
        - **BldgType:** Type of dwelling.
        - **HouseStyle:** Style of dwelling.
        - **OverallQual:** Overall material and finish quality.
        - **OverallCond:** Overall condition rating.
        - **YearBuilt:** Original construction date.
        - **YearRemodAdd:** Remodel date.
        - **RoofStyle:** Type of roof.
        - **RoofMatl:** Roof material.
        - **Exterior1st:** Exterior covering on house.
        - **Exterior2nd:** Exterior covering on house (if more than one material).
        - **MasVnrType:** Masonry veneer type.
        - **MasVnrArea:** Masonry veneer area in square feet.
        - **ExterQual:** Exterior material quality.
        - **ExterCond:** Present condition of the material on the exterior.
        - **Foundation:** Type of foundation.
        - **BsmtQual:** Height of the basement.
        - **BsmtCond:** General condition of the basement.
        - **BsmtExposure:** Walkout or garden level basement walls.
        - **BsmtFinType1:** Quality of basement finished area.
        - **BsmtFinSF1:** Type 1 finished square feet.
        - **BsmtFinType2:** Quality of second finished area (if present).
        - **BsmtFinSF2:** Type 2 finished square feet.
        - **BsmtUnfSF:** Unfinished square feet of basement area.
        - **TotalBsmtSF:** Total square feet of basement area.
        - **Heating:** Type of heating.
        - **HeatingQC:** Heating quality and condition.
        - **CentralAir:** Central air conditioning.
        - **Electrical:** Electrical system.
        - **1stFlrSF:** First Floor square feet.
        - **2ndFlrSF:** Second floor square feet.
        - **LowQualFinSF:** Low quality finished square feet (all floors).
        - **GrLivArea:** Above grade (ground) living area square feet.
        - **BsmtFullBath:** Basement full bathrooms.
        - **BsmtHalfBath:** Basement half bathrooms.
        - **FullBath:** Full bathrooms above grade.
        - **HalfBath:** Half baths above grade.
        - **Bedroom:** Number of bedrooms above basement level.
        - **Kitchen:** Number of kitchens.
        - **KitchenQual:** Kitchen quality.
        - **TotRmsAbvGrd:** Total rooms above grade (does not include bathrooms).
        - **Functional:** Home functionality rating.
        - **Fireplaces:** Number of fireplaces.
        - **FireplaceQu:** Fireplace quality.
        - **GarageType:** Garage location.
        - **GarageYrBlt:** Year garage was built.
        - **GarageFinish:** Interior finish of the garage.
        - **GarageCars:** Size of garage in car capacity.
        - **GarageArea:** Size of garage in square feet.
        - **GarageQual:** Garage quality.
        - **GarageCond:** Garage condition.
        - **PavedDrive:** Paved driveway.
        - **WoodDeckSF:** Wood deck area in square feet.
        - **OpenPorchSF:** Open porch area in square feet.
        - **EnclosedPorch:** Enclosed porch area in square feet.
        - **3SsnPorch:** Three season porch area in square feet.
        - **ScreenPorch:** Screen porch area in square feet.
        - **PoolArea:** Pool area in square feet.
        - **PoolQC:** Pool quality.
        - **Fence:** Fence quality.
        - **MiscFeature:** Miscellaneous feature not covered in other categories.
        - **MiscVal:** $Value of miscellaneous feature.
        - **MoSold:** Month Sold.
        - **YrSold:** Year Sold.
        - **SaleType:** Type of sale.
        - **SaleCondition:** Condition of sale.
        """)


# Prediction page content
with tab2:
    if df_train is not None and df_test is not None:
        prediction_page(df_train, df_test)
    else:
        st.warning("Please upload both the training and test datasets to access the Prediction page.")

# EDA page content
with tab3:
    if df_train is not None and df_test is not None:
        eda_page(df_train, df_test)
    else:
        st.warning("Please upload both the training and test datasets to access the EDA page.")

