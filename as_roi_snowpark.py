#!/usr/bin/env python
# coding: utf-8

# # Snowpark For Python -- Advertising Spend and ROI Prediction
# 
# ### Objective
# 
# In this session, we will train a Linear Regression model to predict future ROI (Return On Investment) of variable ad spend budgets across multiple channels including search, video, social media, and email using Snowpark for Python and scikit-learn. 
# 
# In this Notebook, we will:
# 
# * Create a Session object and securely connect to Snowflake
# * Load data from Snowflake table into Snowpark DataFrame
# * Perform Exploratory Data Analysis (EDA) on Snowpark DataFrame
# * Pivot and Join datasets
# * Create a Python Stored Procedure to deploy model training code on Snowflake
# * Create Python Scalar and Vectorized User-Defined Functions (UDF) for inference on new data points based on user input
#   *  *NOTE: The Scalar UDF is called from the Streamlit App. See [Snowpark_Streamlit_Revenue_Prediction.py](Snowpark_Streamlit_Revenue_Prediction.py)*
# * Create Snowflake Tasks to automate data pipelining and model (re)training

# ### Prerequisites
# 
#   - Snowflake account with ACCOUNTADMIN role
#     - Login to your [Snowflake Trial account](https://app.snowflake.com/) with the admin credentials that were created with the account in one browser tab (a role with ORGADMIN privileges). Keep this tab open during the workshop.
#     - Click on the **Billing** on the left side panel
#     - Click on [Terms and Billing](https://app.snowflake.com/terms-and-billing)
#     - Read and accept terms to continue with the workshop
#   - Python 3.8
#   - Create and Activate Conda Environment (OR, use any other Python environment with Python 3.8) 
#     - conda create --name snowpark -c https://repo.anaconda.com/pkgs/snowflake python=3.8
#     - conda activate snowpark
#   - Install Snowpark for Python, Streamlit and other libraries in Conda environment
#     - conda install -c https://repo.anaconda.com/pkgs/snowflake snowflake-snowpark-python pandas notebook scikit-learn cachetools streamlit
#   - Data Preparation
#     - Follow the steps outlined here to create neccesary tables and load data -- https://github.com/Snowflake-Labs/snowpark-python-demos/tree/main/Advertising-Spend-ROI-Prediction#setup
#   - Update [connection.json](connection.json) with your Snowflake account details and credentials
#     - NOTE: For the account parameter, specify your [account identifier](https://docs.snowflake.com/en/user-guide/admin-account-identifier.html) and do not include the _snowflakecomputing.com_ domain name. Snowflake automatically appends this when creating the connection.
# 
# _For comments and feedback, please reach out to dash.desai@snowflake.com | Follow on [Twitter](https://twitter.com/iamontheinet)_ 

# <div style='text-align: center'>
#     <img src="assets/snowpark.png" alt="Snowpark" style="width: 75%;"/>
# </div>

# ### Import Libraries

# In[1]:


# Snowpark for Python
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import IntegerType, StringType, StructType, FloatType, StructField, DateType, Variant
from snowflake.snowpark.functions import udf, sum, col,array_construct,month,year,call_udf,lit
from snowflake.snowpark.version import VERSION
# Misc
import json
import pandas as pd
import logging 
logger = logging.getLogger("snowflake.snowpark.session")
logger.setLevel(logging.ERROR)


# ### Establish Secure Connection to Snowflake
# 
# Using the Snowpark API, it’s quick and easy to establish a secure connection between Snowflake and Notebook.
# 
#  *Connection options: Username/Password, MFA, OAuth, Okta, SSO*

# In[2]:


# Create Snowflake Session object
connection_parameters = json.load(open('connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

snowflake_environment = session.sql('select current_user(), current_role(), current_database(), current_schema(), current_version(), current_warehouse()').collect()
snowpark_version = VERSION

# Current Environment Details
print('User                        : {}'.format(snowflake_environment[0][0]))
print('Role                        : {}'.format(snowflake_environment[0][1]))
print('Database                    : {}'.format(snowflake_environment[0][2]))
print('Schema                      : {}'.format(snowflake_environment[0][3]))
print('Warehouse                   : {}'.format(snowflake_environment[0][5]))
print('Snowflake version           : {}'.format(snowflake_environment[0][4]))
print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))


# ### Load Aggregated Campaign Spend Data from Snowflake table into Snowpark DataFrame
# 
# Let's first load the campaign spend data. This table contains ad click data that has been aggregated to show daily spend across digital ad channels including search engines, social media, email and video.
# 
# NOTE: Ways to load data in a Snowpark Dataframe
# * session.table("db.schema.table")
# * session.sql("select col1, col2... from tableName")
# * session.read.parquet("@stageName/path/to/file")
# * session.create_dataframe([1,2,3], schema=["col1"])
# 
# TIP: For more information on Snowpark DataFrames, refer to the [docs](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/_autosummary/snowflake.snowpark.html#snowflake.snowpark.DataFrame).
# 

# In[3]:


snow_df_spend = session.table('campaign_spend')
snow_df_spend.queries


# <div style='text-align: center'>
#     <img src="assets/snowpark_python_api.png" alt="Snowpark" style="width: 75%;"/>
# </div>

# In[4]:


# Action sends the DF SQL for execution
# Note: history object provides the query ID which can be helpful for debugging as well as the SQL query executed on the server
with session.query_history() as history:
    snow_df_spend.show()
history.queries


# ### Total Spend per Channel per Month
# 
# Let's transform the data so we can see total cost per year/month per channel using _group_by()_ and _agg()_ Snowpark DataFrame functions.
# 
# TIP: For a full list of functions, refer to the [docs](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/_autosummary/snowflake.snowpark.functions.html#module-snowflake.snowpark.functions).

# In[5]:


# Stats per Month per Channel
snow_df_spend_per_channel = snow_df_spend.group_by(year('DATE'), month('DATE'),'CHANNEL').agg(sum('TOTAL_COST').as_('TOTAL_COST')).\
    with_column_renamed('"YEAR(DATE)"',"YEAR").with_column_renamed('"MONTH(DATE)"',"MONTH").sort('YEAR','MONTH')

snow_df_spend_per_channel.show(10)


# ### Pivot on Channel
# 
#  Let's further transform the campaign spend data so that **each row will represent total cost across all channels** per year/month using _pivot()_ and _sum()_ Snowpark DataFrame functions. This transformation will enable us to join with the revenue table such that we will have our input features and target variable in a single table for model training. 
# 
#  TIP: For a full list of functions, refer to the [docs](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/_autosummary/snowflake.snowpark.functions.html#module-snowflake.snowpark.functions).

# In[6]:


snow_df_spend_per_month = snow_df_spend_per_channel.pivot('CHANNEL',['search_engine','social_media','video','email']).sum('TOTAL_COST').sort('YEAR','MONTH')
snow_df_spend_per_month = snow_df_spend_per_month.select(
    col("YEAR"),
    col("MONTH"),
    col("'search_engine'").as_("SEARCH_ENGINE"),
    col("'social_media'").as_("SOCIAL_MEDIA"),
    col("'video'").as_("VIDEO"),
    col("'email'").as_("EMAIL")
)
snow_df_spend_per_month.show()


# ### Total Revenue per Month

# Now let's load revenue table and transform the data into revenue per year/month using _group_by_ and _agg()_ functions.

# In[7]:


snow_df_revenue = session.table('monthly_revenue')
snow_df_revenue_per_month = snow_df_revenue.group_by('YEAR','MONTH').agg(sum('REVENUE')).sort('YEAR','MONTH').with_column_renamed('SUM(REVENUE)','REVENUE')
snow_df_revenue_per_month.show()


# ### Join Total Spend and Total Revenue per Month

# Next let's **join this revenue data with the transformed campaign spend data** so that our input features (i.e. cost per channel) and target variable (i.e. revenue) can be loaded into a single table for model training. 

# In[8]:


snow_df_spend_and_revenue_per_month = snow_df_spend_per_month.join(snow_df_revenue_per_month, ["YEAR","MONTH"])
snow_df_spend_and_revenue_per_month.show()


# ### >>>>>>>>>> *Examine Snowpark DataFrame Query and Execution Plan* <<<<<<<<<<
# 
# Snowpark makes is really convenient to look at the DataFrame query and execution plan using _explain()_ Snowpark DataFrame function.

# In[9]:


snow_df_spend_and_revenue_per_month.explain()


# ### Model Training in Snowflake 
# 
# #### Features and Target
# 
# At this point we are ready to perform the following actions to save features and target for model training.
# 
# * Delete rows with missing values
# * Exclude columns we don't need for modeling
# * Save features into a Snowflake table called MARKETING_BUDGETS_FEATURES
# 
# TIP: To see how to handle missing values in Snowpark Python, refer to this [blog](https://medium.com/snowflake/handling-missing-values-with-snowpark-for-python-part-1-4af4285d24e6).

# In[10]:


# Delete rows with missing values
snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.dropna()

# Exclude columns we don't need for modeling
snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.drop(['YEAR','MONTH'])

# Save features into a Snowflake table call MARKETING_BUDGETS_FEATURES
snow_df_spend_and_revenue_per_month.write.mode('overwrite').save_as_table('MARKETING_BUDGETS_FEATURES')
snow_df_spend_and_revenue_per_month.show()


# #### Python function to train a Linear Regression model using scikit-learn
# 
# Let's create a Python function that uses **scikit-learn and other packages which are already included in** [Snowflake Anaconda channel](https://repo.anaconda.com/pkgs/snowflake/) and therefore available on the server-side when executing the Python function as a Stored Procedure running in Snowflake.
# 
# This function takes the following as parameters:
# 
# * _session_: Snowflake Session object.
# * _features_table_: Name of the table that holds the features and target variable.
# * _number_of_folds_: Number of cross validation folds used in GridSearchCV.
# * _polynomial_features_degress_: PolynomialFeatures as a preprocessing step.
# * _train_accuracy_threshold_: Accuracy thresholds for train dataset. This values is used to determine if the model should be saved.
# * _test_accuracy_threshold_: Accuracy thresholds for test dataset. This values is used to determine if the model should be saved.
# * _save_model_: Boolean that determines if the model should be saved provided the accuracy thresholds are met.
# 
# TIP: For large datasets, Snowflake offers [Snowpark-optimized Warehouses](https://docs.snowflake.com/en/user-guide/warehouses-snowpark-optimized.html) which are in Public Preview as of Nov 2022).
# 
# 

# In[11]:


def train_revenue_prediction_model(
    session: Session, 
    features_table: str, 
    number_of_folds: int, 
    polynomial_features_degrees: int, 
    train_accuracy_threshold: float, 
    test_accuracy_threshold: float, 
    save_model: bool) -> Variant:
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, GridSearchCV

    import os
    from joblib import dump

    # Load features
    df = session.table(features_table).to_pandas()

    # Preprocess the Numeric columns
    # We apply PolynomialFeatures and StandardScaler preprocessing steps to the numeric columns
    # NOTE: High degrees can cause overfitting.
    numeric_features = ['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL']
    numeric_transformer = Pipeline(steps=[('poly',PolynomialFeatures(degree = polynomial_features_degrees)),('scaler', StandardScaler())])

    # Combine the preprocessed step together using the Column Transformer module
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    # The next step is the integrate the features we just preprocessed with our Machine Learning algorithm to enable us to build a model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LinearRegression())])
    parameteres = {}

    X = df.drop('REVENUE', axis = 1)
    y = df['REVENUE']

    # Split dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    # Use GridSearch to find the best fitting model based on number_of_folds folds
    model = GridSearchCV(pipeline, param_grid=parameteres, cv=number_of_folds)

    model.fit(X_train, y_train)
    train_r2_score = model.score(X_train, y_train)
    test_r2_score = model.score(X_test, y_test)

    model_saved = False

    if save_model:
        if train_r2_score >= train_accuracy_threshold and test_r2_score >= test_accuracy_threshold:
            # Upload trained model to a stage
            model_output_dir = '/tmp'
            model_file = os.path.join(model_output_dir, 'model.joblib')
            dump(model, model_file)
            session.file.put(model_file,"@dash_models",overwrite=True)
            model_saved = True

    # Return model R2 score on train and test data
    return {"R2 score on Train": train_r2_score,
            "R2 threshold on Train": train_accuracy_threshold,
            "R2 score on Test": test_r2_score,
            "R2 threshold on Test": test_accuracy_threshold,
            "Model saved": model_saved}


# #### Test Python function before deploying it as a Stored Procedure on Snowflake
# 
# Since we're in test mode, we will set _save_model = False_ so that the model is not saved just yet.

# In[12]:


cross_validaton_folds = 10
polynomial_features_degrees = 2
train_accuracy_threshold = 0.85
test_accuracy_threshold = 0.85
save_model = False

train_revenue_prediction_model(
    session,
    "MARKETING_BUDGETS_FEATURES",
    cross_validaton_folds,
    polynomial_features_degrees,
    train_accuracy_threshold,
    test_accuracy_threshold,
    save_model)


# ### Create Stored Procedure to deploy model training code on Snowflake
# 
# Assuming the testing is complete and we're satisfied with the model, let's **register the model training Python function as a Snowpark Python Stored Procedure** by supplying the packages (_snowflake-snowpark-python,scikit-learn, and joblib_) it will need and use during execution.
# 
# TIP: For more information on Snowpark Python Stored Procedures, refer to the [docs](https://docs.snowflake.com/en/sql-reference/stored-procedures-python.html).

# In[13]:


session.sproc.register(
    func=train_revenue_prediction_model,
    name="train_revenue_prediction_model",
    packages=['snowflake-snowpark-python','scikit-learn','joblib'],
    is_permanent=True,
    stage_location="@dash_sprocs",
    replace=True)


# ### >>>>>>>>>> *Examine Query History in Snowsight* <<<<<<<<<<

# ### Execute Stored Procedure to train model and deploy it on Snowflake
# 
# Now we're ready to train the model and save it onto a Snowflake stage so let's set _save_model = True_ and run/execute the Stored Procedure using _session.call()_ function.

# In[14]:


cross_validaton_folds = 10
polynomial_features_degrees = 2
train_accuracy_threshold = 0.85
test_accuracy_threshold = 0.85
save_model = True

print(session.call('train_revenue_prediction_model',
                    'MARKETING_BUDGETS_FEATURES',
                    cross_validaton_folds,
                    polynomial_features_degrees,
                    train_accuracy_threshold,
                    test_accuracy_threshold,
                    save_model))


# ### >>>>>>>>>> *Examine Query History in Snowsight* <<<<<<<<<<

# ### Create Scalar User-Defined Function (UDF) for inference
# 
# Now to deploy this model for inference, let's **create and register a Snowpark Python UDF and add the trained model as a dependency**. Once registered, getting new predictions is as simple as calling the function by passing in data.
# 
# *NOTE: Scalar UDFs operate on a single row / set of data points and are great for online inference in real-time. And this UDF is called from the Streamlit App. See [Snowpark_Streamlit_Revenue_Prediction.py](Snowpark_Streamlit_Revenue_Prediction.py)*
# 
# TIP: For more information on Snowpark Python User-Defined Functions, refer to the [docs](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-udfs.html).

# In[15]:


session.clear_imports()
session.clear_packages()

# Add trained model and Python packages from Snowflake Anaconda channel available on the server-side as UDF dependencies
session.add_import('@dash_models/model.joblib.gz')
session.add_packages('pandas','joblib','scikit-learn==1.1.1')

@udf(name='predict_roi',session=session,replace=True,is_permanent=True,stage_location='@dash_udfs')
def predict_roi(budget_allocations: list) -> float:
    import sys
    import pandas as pd
    from joblib import load
    import sklearn

    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    
    model_file = import_dir + 'model.joblib.gz'
    model = load(model_file)
            
    features = ['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL']
    df = pd.DataFrame([budget_allocations], columns=features)
    roi = abs(model.predict(df)[0])
    return roi


# ### Call Scalar User-Defined Function (UDF) for inference on new data

#  Once the UDF is registered, getting new predictions is as simple as calling the _call_udf()_ Snowpark Python function and passing in new datapoints.
# 
# Let's create a SnowPark DataFrame with some sample data and call the UDF to get new predictions.
# 
#  *NOTE: This UDF is also called from the Streamlit App. See [Snowpark_Streamlit_Revenue_Prediction.py](Snowpark_Streamlit_Revenue_Prediction.py)*

# In[16]:


test_df = session.create_dataframe([[250000,250000,200000,450000],[500000,500000,500000,500000],[8500,9500,2000,500]], 
                                    schema=['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL'])
test_df.select(
    'SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL', 
    call_udf("predict_roi", 
    array_construct(col("SEARCH_ENGINE"), col("SOCIAL_MEDIA"), col("VIDEO"), col("EMAIL"))).as_("PREDICTED_ROI")).show()


# ### Create Vectorized User-Defined Function (UDF) using Batch API for inference
# 
# Here we will leverage the Python UDF Batch API to create a **vectorized** UDF which takes a Pandas Dataframe as input. This means that each call to the UDF receives a set/batch of rows compared to a Scalar UDF which gets one row as input. 
# 
# First we will create a helper function _load_model()_ that uses **cachetools** to make sure we only load the model once followed by _batch_predict_roi()_ function that does the inference. 
# 
# _NOTE: Vectorized UDFs are great for offline inference in batch mode._
# 
# Advantages of using the Batch API over Scalar UDFs:
# 
# * The potential for better performance if your Python code operates efficiently on batches of rows
# * Less transformation logic required if you are calling into libraries that operate on Pandas DataFrames or Pandas arrays
# 
# TIP: For more information on Snowpark Python UDF Batch API, refer to the [docs](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-batch.html#getting-started-with-the-batch-api).

# In[17]:


session.clear_imports()
session.clear_packages()

import cachetools
from snowflake.snowpark.types import PandasSeries, PandasDataFrame

# Add trained model and Python packages from Snowflake Anaconda channel available on the server-side as UDF dependencies
session.add_import('@dash_models/model.joblib.gz')
session.add_packages('pandas','joblib','scikit-learn','cachetools')

@cachetools.cached(cache={})
def load_model(filename):
    import joblib
    import sys
    import os

    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]

    if import_dir:
        with open(os.path.join(import_dir, filename), 'rb') as file:
            m = joblib.load(file)
            return m

@udf(name='batch_predict_roi',session=session,replace=True,is_permanent=True,stage_location='@dash_udfs')
def batch_predict_roi(budget_allocations_df: PandasDataFrame[int, int, int, int]) -> PandasSeries[float]:
    import sklearn
    budget_allocations_df.columns = ['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL']
    model = load_model('model.joblib.gz')
    return abs(model.predict(budget_allocations_df))


# ### Call Vectorized User-Defined Function (UDF) using Batch API for inference on new data
# 
# When you use the Batch API:
# 
# * You do not need to change how you write queries using Python UDFs. All batching is handled by the UDF framework rather than your own code
# * NOTE: As with the non-batch / scalar API, there is no guarantee of which instances of your handler code will see which batches of input

# In[18]:


test_df.select(
    'SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL', 
    call_udf("batch_predict_roi", 
    col("SEARCH_ENGINE"), col("SOCIAL_MEDIA"), col("VIDEO"), col("EMAIL")).as_("PREDICTED_ROI")).show()


# **Snowpark Stored Procedures vs User-Defined Functions**
# 
# _In general, if you're processing a large dataset in a way where each row/batch can be processed independently - UDFs are always better, because the processing is automatically parallelized/scaled across the warehouse. For example, if you already have a trained ML model, and you're doing inference using that model on billions of rows. In that case, each row/batch can be computed independently._
# 
# _If the use case requires the full dataset to be in-memory (e.g. ML training), then a stored procedure is the way to go. A stored procedure is just a Python program that runs on a single warehouse node. (With a UDF it's not possible to load the full dataset into memory because the processing is done in a streaming fashion, one batch at a time._

# ### Automate Data Pipeline and Model (re)Training using Snowflake Tasks
# 
# We can also optionally create Snowflake (Serverless or User-managed) Tasks to automate data pipelining and (re)training of the model on a set schedule.
# 
# _NOTE: Creating tasks using Snowpark Python API (instead of SQL) is on the roadmap. Stay tuned! Or, follow me on [Twitter]((https://twitter.com/iamontheinet)_) to get the news before anyone else :)_
# 
# TIP: Amongst other things, you can also configure tasks for error notification (currently in Private Preview) using cloud messaging service such as Amazon Simple Notification Service (SNS). For more information on Snowflake Tasks, refer to the [docs](https://docs.snowflake.com/en/user-guide/tasks-intro.html).

# #### Create Python Function for Data Pipeline and Feature Engineering 

# In[19]:


def data_pipeline_feature_engineering(session: Session) -> str:

  # DATA TRANSFORMATIONS
  # Perform the following actions to transform the data

  # Load the campaign spend data
  snow_df_spend = session.table('campaign_spend')

  # Transform the data so we can see total cost per year/month per channel using group_by() and agg() Snowpark DataFrame functions
  snow_df_spend_per_channel = snow_df_spend.group_by(year('DATE'), month('DATE'),'CHANNEL').agg(sum('TOTAL_COST').as_('TOTAL_COST')).\
      with_column_renamed('"YEAR(DATE)"',"YEAR").with_column_renamed('"MONTH(DATE)"',"MONTH").sort('YEAR','MONTH')

  # Transform the data so that each row will represent total cost across all channels per year/month using pivot() and sum() Snowpark DataFrame functions
  snow_df_spend_per_month = snow_df_spend_per_channel.pivot('CHANNEL',['search_engine','social_media','video','email']).sum('TOTAL_COST').sort('YEAR','MONTH')
  snow_df_spend_per_month = snow_df_spend_per_month.select(
      col("YEAR"),
      col("MONTH"),
      col("'search_engine'").as_("SEARCH_ENGINE"),
      col("'social_media'").as_("SOCIAL_MEDIA"),
      col("'video'").as_("VIDEO"),
      col("'email'").as_("EMAIL")
  )

  # Load revenue table and transform the data into revenue per year/month using group_by and agg() functions
  snow_df_revenue = session.table('monthly_revenue')
  snow_df_revenue_per_month = snow_df_revenue.group_by('YEAR','MONTH').agg(sum('REVENUE')).sort('YEAR','MONTH').with_column_renamed('SUM(REVENUE)','REVENUE')

  # Join revenue data with the transformed campaign spend data so that our input features (i.e. cost per channel) and target variable (i.e. revenue) can be loaded into a single table for model training
  snow_df_spend_and_revenue_per_month = snow_df_spend_per_month.join(snow_df_revenue_per_month, ["YEAR","MONTH"])

  # SAVE FEATURES And TARGET
  # Perform the following actions to save features and target for model training

  # Delete rows with missing values
  snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.dropna()

  # Exclude columns we don't need for modeling
  snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.drop(['YEAR','MONTH'])

  # Save features into a Snowflake table call MARKETING_BUDGETS_FEATURES
  snow_df_spend_and_revenue_per_month.write.mode('overwrite').save_as_table('MARKETING_BUDGETS_FEATURES')

  return "SUCCESS"


# #### Create Stored Procedure to deploy data pipelining feature engineeering code on Snowflake
# 
# TIP: For more information on Snowpark Python Stored Procedures, refer to the [docs](https://docs.snowflake.com/en/sql-reference/stored-procedures-python.html).

# In[20]:


session.sproc.register(
    func=data_pipeline_feature_engineering,
    name="data_pipeline_feature_engineering",
    packages=['snowflake-snowpark-python'],
    is_permanent=True,
    stage_location="@dash_sprocs",
    replace=True)


# #### Execute Stored Procedure to deploy data pipelining feature engineeering code on Snowflake

# In[21]:


print(session.call('data_pipeline_feature_engineering'))


# #### Create Root/Parent Snowflake Task: Data pipelining and feature engineeering

# In[23]:


create_data_pipeline_feature_engineering_task = """
CREATE OR REPLACE TASK data_pipeline_feature_engineering_task
    WAREHOUSE = 'AS_ROI_WH'
    SCHEDULE  = '1 MINUTE'
AS
    CALL data_pipeline_feature_engineering()
"""
session.sql(create_data_pipeline_feature_engineering_task).collect()


# #### Create Child/Dependent Snowflake Task: Model training on Snowflake

# In[24]:


create_model_training_task = """
CREATE OR REPLACE TASK model_training_task
    WAREHOUSE = 'AS_ROI_WH'
    AFTER data_pipeline_feature_engineering_task
AS
    CALL train_revenue_prediction_model('MARKETING_BUDGETS_FEATURES',10,2,0.85,0.85,True)
"""
session.sql(create_model_training_task).collect()


# #### Resume Tasks

# In[25]:


session.sql("alter task model_training_task resume").collect()
session.sql("alter task data_pipeline_feature_engineering_task resume").collect()


# #### Cleanup Resources

# In[26]:


session.sql("alter task data_pipeline_feature_engineering_task suspend").collect()
session.sql("alter task model_training_task suspend").collect()

