import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import tensorflow as tf
from tensorflow import keras
import json
import os

print("🧪 Starting Water-Borne Disease Model Training...")

# Create realistic synthetic dataset for water-borne diseases
np.random.seed(42)
n_samples = 10000

print("📊 Creating synthetic dataset...")

data = {
    'age': np.random.randint(1, 80, n_samples),
    'symptom_duration': np.random.randint(1, 21, n_samples),
    'fever': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    'diarrhea': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'vomiting': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'abdominal_pain': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    'dehydration': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    'blood_in_stool': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'fatigue': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'nausea': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'muscle_cramps': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'headache': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'water_source': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    'severity': np.random.randint(1, 11, n_samples)
}

df = pd.DataFrame(data)

# Create target variable (disease) based on realistic symptom patterns
def assign_disease(row):
    # Cholera: Severe diarrhea, vomiting, dehydration (acute)
    if (row['diarrhea'] == 1 and row['vomiting'] == 1 and 
        row['dehydration'] == 1 and row['symptom_duration'] <= 3 and row['severity'] >= 8):
        return 0  # Cholera
    
    # Dysentery: Blood in stool, abdominal pain, fever
    elif (row['blood_in_stool'] == 1 and row['abdominal_pain'] == 1 and 
          row['fever'] == 1 and row['severity'] >= 6):
        return 1  # Dysentery
    
    # Typhoid: Prolonged fever, headache, fatigue
    elif (row['fever'] == 1 and row['headache'] == 1 and row['fatigue'] == 1 and 
          row['symptom_duration'] >= 7 and row['severity'] >= 5):
        return 2  # Typhoid
    
    # Gastroenteritis: Nausea, vomiting, diarrhea, abdominal pain
    elif (row['nausea'] == 1 and row['vomiting'] == 1 and 
          row['diarrhea'] == 1 and row['abdominal_pain'] == 1):
        return 3  # Gastroenteritis
    
    # Hepatitis A: Fever, fatigue, nausea, muscle cramps
    elif (row['fever'] == 1 and row['fatigue'] == 1 and 
          row['nausea'] == 1 and row['muscle_cramps'] == 1 and row['symptom_duration'] >= 5):
        return 4  # Hepatitis A
    
    else:
        return 5  # No specific disease

df['disease'] = df.apply(assign_disease, axis=1)

print(f"✅ Dataset created with {n_samples} samples")

# Disease mapping
disease_names = {
    0: 'Cholera',
    1: 'Dysentery', 
    2: 'Typhoid',
    3: 'Gastroenteritis',
    4: 'Hepatitis A',
    5: 'No Specific Disease'
}

print("\n🔍 Disease distribution:")
for code, name in disease_names.items():
    count = (df['disease'] == code).sum()
    percentage = (count / n_samples) * 100
    print(f"   {code}: {name} - {count} samples ({percentage:.1f}%)")

# Prepare features and target
X = df.drop('disease', axis=1)
y = df['disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📈 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train Random Forest model (for reference)
print("\n🤖 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_accuracy = rf_model.score(X_test, y_test)
print(f"✅ Random Forest Test Accuracy: {rf_accuracy:.4f}")

# Create output directory
output_dir = 'android_model'
os.makedirs(output_dir, exist_ok=True)

print(f"\n🔄 Creating Neural Network model for TensorFlow Lite...")

# Create a simple neural network that mimics Random Forest behavior
def create_simple_nn(input_dim, output_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train the neural network
nn_model = create_simple_nn(X_train.shape[1], len(disease_names))

print("🧠 Training Neural Network...")
history = nn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate neural network
nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Neural Network Test Accuracy: {nn_accuracy:.4f}")

# Convert to TensorFlow Lite
print("\n📱 Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_path = os.path.join(output_dir, 'disease_predictor.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ TensorFlow Lite model saved as '{tflite_path}'")

# Save the scikit-learn model (for reference)
joblib.dump(rf_model, os.path.join(output_dir, 'disease_predictor_rf_model.pkl'))
print("✅ Random Forest model saved")

# Save dataset
df.to_csv(os.path.join(output_dir, 'waterborne_disease_dataset.csv'), index=False)
print("✅ Dataset saved")

# Create metadata for Android app
metadata = {
    'feature_columns': X.columns.tolist(),
    'disease_mapping': disease_names,
    'input_shape': [1, len(X.columns)],
    'output_classes': list(disease_names.values()),
    'model_accuracy': float(nn_accuracy),
    'model_type': 'neural_network'
}

with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model metadata saved")

# Test with example cases
print("\n🧪 Testing with example cases:")
test_cases = [
    # Cholera case
    [35, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 9],
    # Typhoid case  
    [28, 10, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 6],
    # Gastroenteritis case
    [42, 3, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 7],
    # Healthy case
    [25, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
]

case_descriptions = [
    "Cholera (severe diarrhea + vomiting + dehydration)",
    "Typhoid (prolonged fever + headache + fatigue)",
    "Gastroenteritis (nausea + vomiting + diarrhea)",
    "Healthy (no significant symptoms)"
]

print("🔍 Neural Network Predictions:")
for i, (case, desc) in enumerate(zip(test_cases, case_descriptions)):
    # Convert to numpy array and reshape for NN
    case_array = np.array(case, dtype=np.float32).reshape(1, -1)
    prediction_proba = nn_model.predict(case_array, verbose=0)[0]
    prediction = np.argmax(prediction_proba)
    confidence = prediction_proba[prediction]
    
    print(f"Case {i+1} ({desc}):")
    print(f"  → Predicted: {disease_names[prediction]}")
    print(f"  → Confidence: {confidence:.2%}")
    print(f"  → All probabilities: {[f'{p:.2%}' for p in prediction_proba]}")

print("\n" + "="*70)
print("🎉 MODEL TRAINING COMPLETE!".center(70))
print("="*70)
print(f"\n📁 All files saved in './{output_dir}/' folder:")
print("   1. disease_predictor.tflite     - TensorFlow Lite model (for Android)")
print("   2. model_metadata.json          - Model configuration")

print("\n📱 Android Integration Steps:")
print("   1. Create 'assets' folder in: app/src/main/assets/")
print("   2. Copy the 2 files above to the assets folder")
print("   3. Add TensorFlow Lite dependency to build.gradle")
print("   4. Implement TFLite interpreter in HealthReportFragment")

print(f"\n📍 Output folder: {os.path.abspath(output_dir)}")