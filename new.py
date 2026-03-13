import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Load data function
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# Preprocess the data
def preprocess_data(train_df, test_df):
    # Extract features and labels from training data
    X_train = train_df.iloc[:, 1:-1].values  # assuming features start from second column and end at second-last column
    y_train = train_df.iloc[:, -1].values  # assuming labels are in the last column
    X_test = test_df.iloc[:, 1:].values[:, :X_train.shape[1]]  # Align test features with train features

    # Debugging: Check if the data is loaded properly
    print("Shape of X_train:", X_train.shape)
    print("First few rows of X_train:", X_train[:5])
    print("Shape of y_train:", y_train.shape)
    print("First few labels of y_train:", y_train[:5])

    # Label encoding for categorical labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Check if y_train is encoded properly
    print("Encoded y_train:", y_train[:5])

    # Standard scaling of the feature data (X_train and X_test)
    scaler = StandardScaler()
    
    # Check if the features are empty before scaling
    if X_train.shape[0] > 0:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # Use the same scaler for test data
        print("Scaling completed. X_train and X_test shapes:", X_train.shape, X_test.shape)
    else:
        print("Error: X_train is empty.")
    
    return X_train[..., np.newaxis], y_train, X_test[..., np.newaxis]

# Build the CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 5, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1,
                     callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
                                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Predict labels and print them
def predict_labels(model, X_test, label_encoder, test_df):
    preds = label_encoder.inverse_transform(np.argmax(model.predict(X_test), axis=1))
    
    # Debugging: Check predictions
    for filename, label in zip(test_df.iloc[:, 0], preds):
        print(f"📌 **{filename} → {label}**")

# Main execution
if __name__ == "__main__":
    # Load the datasets
    train_df, test_df = load_data("path_to_trainset.csv", "path_to_testset.csv")
    
    # Preprocess the data
    X_train, y_train, X_test = preprocess_data(train_df, test_df)
    
    # Build and train the model
    model = build_cnn((X_train.shape[1], 1), len(np.unique(y_train)))
    history = train_model(model, X_train, y_train)
    
    # Print final training and validation accuracy
    print(f"✅ Training Accuracy: {history.history['accuracy'][-1]:.4f} | Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Predict labels for the test set
    predict_labels(model, X_test, LabelEncoder().fit(train_df.iloc[:, -1]), test_df)
