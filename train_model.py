import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Sample Dataset
data = {
    'StudyHours': [2, 5, 8, 3, 6, 1, 4, 7],
    'Attendance': [70, 85, 95, 75, 90, 60, 80, 92],
    'PreviousGrade': ['C', 'B', 'A', 'C', 'B', 'D', 'C', 'A'],
    'Performance': ['Poor', 'Average', 'Good', 'Poor', 'Good', 'Poor', 'Average', 'Good']
}
df = pd.DataFrame(data)

# Step 2: Encode Categorical Columns
label_encoder_grade = LabelEncoder()
label_encoder_performance = LabelEncoder()

df['PreviousGrade'] = label_encoder_grade.fit_transform(df['PreviousGrade'])  # A=0, B=1...
df['Performance'] = label_encoder_performance.fit_transform(df['Performance'])  # Poor=1, Average=0, Good=2

# Step 3: Split Data
X = df[['StudyHours', 'Attendance', 'PreviousGrade']]
y = df['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: Save Model
with open('model_performance.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved as model_performance.pkl")
