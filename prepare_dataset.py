import pandas as pd

# Define the headers based on the features you provided
headers = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
headers = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm"
]
# Read the adult.data file without headers
data = pd.read_csv('data/adult.data', header=None, names=headers, na_values=' ?')
data2 = pd.read_csv('data/adult.test', header=None, names=headers, na_values=' ?')
data = pd.read_csv('data/iris.data', header=None, names=headers, na_values=' ?')
# Concatenate the two dataframes
#data = pd.concat([data, data2], ignore_index=True)
# Save the data to a new CSV file with headers
data.to_csv('iris_with_headers.csv', index=False)

print("CSV file with headers has been created successfully.")
