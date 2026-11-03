from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()

x = iris.data
y = iris.target

model = RandomForestClassifier()

model.fit(x, y)

joblib.dump(model, "iris_model.pkl")

print("Model trained and saved successfully!")
