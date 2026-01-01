import streamlit as st
import pandas as pd
import pickle

# Load model and preprocessor
with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.title("House Price Predictor 🏠")
st.header("Enter property details:")

# -------------------------------
# Numeric inputs
# -------------------------------
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
sqft_living = st.number_input("Sqft Living", min_value=100, max_value=10000, value=1500)
sqft_lot = st.number_input("Sqft Lot", min_value=100, max_value=100000, value=5000)
floors = st.number_input("Floors", min_value=1, max_value=10, value=1)
waterfront = st.number_input("Waterfront (0=no,1=yes)", min_value=0, max_value=1, value=0)
view = st.number_input("View (0-4)", min_value=0, max_value=4, value=0)
condition = st.number_input("Condition (1-5)", min_value=1, max_value=5, value=3)
sqft_above = st.number_input("Sqft Above", min_value=100, max_value=10000, value=1500)
sqft_basement = st.number_input("Sqft Basement", min_value=0, max_value=5000, value=0)

# Year built and renovated as dropdowns

years = list(range(1800, 2027))  # Or dynamically get current year
yr_built = st.selectbox("Year Built", options=years, index=years.index(1990))

years_renovated = [0] + years
yr_renovated = st.selectbox("Year Renovated (0 if none)", options=years_renovated, index=0)


# -------------------------------
# Categorical inputs as dropdowns
# -------------------------------

city_options = ['Shoreline', 'Kent', 'Bellevue', 'Redmond', 'Seattle', 'Maple Valley',
                'North Bend', 'Lake Forest Park', 'Sammamish', 'Auburn', 'Des Moines',
                'Bothell', 'Federal Way', 'Kirkland', 'Issaquah', 'Woodinville',
                'Normandy Park', 'Fall City', 'Renton', 'Carnation', 'Snoqualmie', 'Duvall',
                'Burien', 'Covington', 'Inglewood-Finn Hill', 'Kenmore', 'Newcastle',
                'Black Diamond', 'Ravensdale', 'Clyde Hill', 'Algona', 'Mercer Island',
                'Skykomish', 'Tukwila', 'Vashon', 'SeaTac', 'Enumclaw', 'Snoqualmie Pass',
                'Pacific', 'Beaux Arts Village', 'Preston', 'Milton', 'Yarrow Point', 'Medina']

city = st.selectbox("City", options=city_options)

country_options = ['USA']  # Only one country
country = st.selectbox("Country", options=country_options)

state_options = ['WA']  # Only one state in your dataset
state = st.selectbox("State", options=state_options)

zip_options = ['98133', '98042', '98008', '98052', '98115', '98038', '98045', '98155',
               '98074', '98106', '98007', '98092', '98198', '98006', '98102', '98011',
               '98125', '98003', '98136', '98033', '98029', '98117', '98034', '98072',
               '98023', '98107', '98166', '98116', '98024', '98055', '98077', '98027',
               '98059', '98075', '98014', '98065', '98199', '98053', '98058', '98122',
               '98103', '98112', '98005', '98118', '98177', '98105', '98004', '98019',
               '98119', '98144', '98168', '98001', '98056', '98146', '98028', '98148',
               '98057', '98010', '98051', '98031', '98030', '98126', '98032', '98178',
               '98040', '98288', '98108', '98070', '98109', '98188', '98002', '98022',
               '98068', '98047', '98050', '98354', '98039']

zip_code = st.selectbox("ZIP Code", options=zip_options)


# -------------------------------
# Make prediction
# -------------------------------
if st.button("Predict Price"):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'condition': [condition],
        'sqft_above': [sqft_above],
        'sqft_basement': [sqft_basement],
        'yr_built': [yr_built],
        'yr_renovated': [yr_renovated],
        'city': [city],
        'country': [country],
        'state': [state],
        'zip': [zip_code]
    })

    # Preprocess input
    input_processed = preprocessor.transform(input_df)

    # Predict
    predicted_price = model.predict(input_processed)

    st.success(f"Predicted House Price: ${predicted_price[0]:,.2f}")
