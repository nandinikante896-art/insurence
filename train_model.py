import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# load dataset
df = pd.read_csv("insurance.csv")

# remove spaces in column names
df.columns = df.columns.str.strip()

# convert categorical values
df["sex"] = df["sex"].map({"female":0, "male":1})
df["smoker"] = df["smoker"].map({"no":0, "yes":1})
df["region"] = df["region"].map({
    "southwest":0,
    "southeast":1,
    "northwest":2,
    "northeast":3
})

# features and target
X = df.drop("expenses", axis=1)
y = df["expenses"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "insurance_model.pkl")

print("Model trained and saved successfully")