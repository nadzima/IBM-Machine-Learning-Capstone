import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup - Ensure this is the first Streamlit command
st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Background for the entire app */
    .reportview-container {
        background-image: url('https://example.com/background.jpg'); /* Replace with your background image URL */
        background-size: cover;
        background-attachment: fixed;
        color: #333; /* Default text color */
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: #ecf0f1;
    }

    .sidebar .sidebar-content h1 {
        color: #ecf0f1;
    }

    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #2980b9;
    }

    /* Selectbox and Slider styling */
    .stSelectbox>div, .stSlider>div {
        border-radius: 5px;
        background-color: #ecf0f1;
        color: #333;
    }

    /* DataTable styling */
    .stDataFrame {
        margin-top: 20px;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ------- Functions ------
# Load datasets with new caching methods
@st.cache_data
def load_ratings():
    return backend.load_ratings()

@st.cache_data
def load_course_sims():
    return backend.load_course_sims()

@st.cache_data
def load_courses():
    return backend.load_courses()

@st.cache_data
def load_bow():
    return backend.load_bow()

# Initialize the app by loading datasets
def init_recommender_app():
    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
    
    st.success('Datasets loaded successfully!')
    st.markdown("---")
    st.subheader("Select courses you have audited or completed:")

    # Interactive table for `course_df` with sorting enabled
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True, enableSorting=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your selected courses:")
    st.dataframe(results, use_container_width=True)
    return results

def train(model_name, params):
    if model_name == backend.models[0]:
        with st.spinner('Training model...'):
            time.sleep(0.5)  # Simulate training
            backend.train(model_name)
        st.success('Training complete!')
    # TODO: Add other model training code here
    elif model_name == backend.models[1]:
        pass
    else:
        pass

def predict(model_name, user_ids, params):
    with st.spinner('Generating course recommendations...'):
        time.sleep(0.5)  # Simulate prediction
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')
    return res

# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
selected_courses_df = init_recommender_app()

# Model selection
st.sidebar.subheader('1. Select Recommendation Model')
model_selection = st.sidebar.selectbox(
    "Choose model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters')
if model_selection == backend.models[0]:
    top_courses = st.sidebar.slider('Top Courses', min_value=0, max_value=100, value=10, step=1)
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold (%)', min_value=0, max_value=100, value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold (%)', min_value=0, max_value=100, value=50, step=10)
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters', min_value=0, max_value=50, value=20, step=1)

# Training
st.sidebar.subheader('3. Train Model')
if st.sidebar.button("Train Model"):
    train(model_selection, params)

# Prediction
st.sidebar.subheader('4. Get Recommendations')
if st.sidebar.button("Recommend New Courses") and selected_courses_df.shape[0] > 0:
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.dataframe(res_df, use_container_width=True)
