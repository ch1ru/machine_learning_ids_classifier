import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
print("Loading raw data")
df_training_raw = pd.read_csv('./CMP_training dataset1.csv')
df_testing_raw  = pd.read_csv('./CMP_testing dataset1.csv')
df_raw = pd.concat([df_training_raw, df_testing_raw], ignore_index=True)

# Preprocess
print("Transforming data")
cols_to_drop = [
    'stcpb', 'dtcpb', 'is_ftp_login', 'ct_ftp_cmd',
    'is_sm_ips_ports', 'tcprtt', 'swin', 'dwin', 'trans_depth',
    'id', 'rate', 'ct_flw_http_mthd', 'response_body_len',
    'synack', 'ackdat', 'dloss', 'sloss',
    'label'                               # drop binary label, keep attack_cat
]
df_raw = df_raw.drop(columns=cols_to_drop)

df_raw['proto'],      protocols = pd.factorize(df_raw['proto'])
df_raw['service'],    services  = pd.factorize(df_raw['service'])
df_raw['state'],      flags     = pd.factorize(df_raw['state'])
df_raw['attack_cat'], attacks   = pd.factorize(df_raw['attack_cat'])

num_classes = len(attacks)
print(f"Classes ({num_classes}): {list(attacks)}")

labels   = df_raw['attack_cat'].values
features = df_raw.drop(columns='attack_cat')

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.8, test_size=0.2, random_state=15, stratify=labels
)
#print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# Scale
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

y_train_ohe = keras.utils.to_categorical(y_train, num_classes)
y_test_ohe  = keras.utils.to_categorical(y_test,  num_classes)

# Build Model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_sc.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(num_classes, activation='softmax')  # softmax for multiclass (use sigmoid for binary classification)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',                       # crossentropy for multiclass (use binary_crossentropy for binary classification)
    metrics=['accuracy']
)

model.summary()

# Train Model
history = model.fit(
    X_train_sc, y_train_ohe,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
    ],
    verbose=1
)

# Loss Curves
def plot_loss_curves(history, title='Loss Curves'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--')
    axes[0].set_title('Loss (Categorical Cross-Entropy)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_loss_curves(history, title='Multiclass Attack Classification for multi-layer perceptron model')

# Evaluate Model
y_pred_prob = model.predict(X_test_sc)
y_pred = np.argmax(y_pred_prob, axis=1)

all_labels = list(range(num_classes))
all_names = list(attacks)

print("\nClassification Report:\n",
      classification_report(y_test, y_pred,
                            labels=all_labels,
                            target_names=all_names,
                            zero_division=0))

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names, all_labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, ax=ax,
        annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5, linecolor='lightgrey'
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()

plot_confusion_matrix(
    y_test, y_pred,
    class_names=all_names,
    all_labels=all_labels,
    title='Neural Network Model: Confusion Matrix for Multiclass Attack Classification'
)
