import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import joblib

# ✅ 1. Load your CSV file
df = pd.read_csv('hotel_bookings.csv')

# ✅ 2. Encode all text columns
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# ✅ 3. Select only the 9 features used in Flask form
selected_features = [
    'country', 
    'deposit_type', 
    'lead_time', 
    'total_of_special_requests', 
    'adr', 
    'market_segment', 
    'arrival_date_day_of_month', 
    'arrival_date_week_number', 
    'stays_in_week_nights'
]

X = df[selected_features]
y = df['is_canceled']

# ✅ 4. Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 5. Drop missing values in both train and test sets
X_train = X_train.dropna()
y_train = y_train[X_train.index]

X_test = X_test.dropna()
y_test = y_test[X_test.index]

# ✅ 6. Balance the training data with SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ✅ 7. Train the model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_sm, y_train_sm)

# ✅ 8. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# ✅ 9. Save the model
joblib.dump(model, 'GBModelNormalData.pkl')
print("✅ Model saved as GBModelNormalData.pkl")
