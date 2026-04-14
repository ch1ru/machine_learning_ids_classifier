import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Data
print("Loading raw data")
df_training_raw = pd.read_csv('./CMP_training dataset1.csv')
df_testing_raw  = pd.read_csv('./CMP_testing dataset1.csv')
df_raw = pd.concat([df_training_raw, df_testing_raw], ignore_index=True)

# Preprocess
print("Transforming data")
cols_to_drop = [
    'stcpb', 'dtcpb', 'is_ftp_login', 'ct_ftp_cmd', 'sttl', 'dttl',
    'is_sm_ips_ports', 'tcprtt', 'swin', 'dwin', 'trans_depth',
    'id', 'rate', 'ct_flw_http_mthd', 'response_body_len',
    'synack', 'ackdat', 'dloss', 'sloss',
    'attack_cat'                               
]
df_raw = df_raw.drop(columns=cols_to_drop)

df_raw['proto'],      protocols = pd.factorize(df_raw['proto'])
df_raw['service'],    services  = pd.factorize(df_raw['service'])
df_raw['state'],      flags     = pd.factorize(df_raw['state'])
df_raw['label'], attacks = pd.factorize(df_raw['label'])

le = LabelEncoder()
df_raw['label'] = le.fit_transform(df_raw['label'])
num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {list(le.classes_)}")

# drop binary labels (note, this is switched depending on if its binary or multiclass classification)
features = df_raw.drop(columns='label')
labels   = df_raw['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.8, test_size=0.2,
    random_state=42, stratify=labels
)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    oob_score=True                        
)
rf.fit(X_train, y_train)

# OOB error curve
oob_errors = []
for n in range(1, rf.n_estimators + 1):
    rf_temp = RandomForestClassifier(
        n_estimators=n,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        oob_score=True,
        warm_start=True                   
    )
    rf_temp.fit(X_train, y_train)
    oob_errors.append(1 - rf_temp.oob_score_)

def plot_rf_curves(oob_errors, train_score, test_score):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Binary (malicious/benign traffic) Attack Classification for Random Forest Model', fontsize=14, fontweight='bold')

    axes[0].plot(range(1, len(oob_errors) + 1), oob_errors,
                 color='steelblue', linewidth=2)
    axes[0].set_title('OOB Error vs Number of Trees')
    axes[0].set_xlabel('Number of Trees')
    axes[0].set_ylabel('OOB Error Rate')
    axes[0].grid(True, alpha=0.3)

    # Train vs Test accuracy bar chart
    axes[1].bar(['Train', 'Test'], [train_score, test_score],
                color=['steelblue', 'coral'], width=0.4)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Train vs Test Accuracy')
    axes[1].set_ylabel('Accuracy')
    for i, v in enumerate([train_score, test_score]):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Feature importances (top 15)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top15 = importances.sort_values(ascending=True).tail(15)
    axes[2].barh(top15.index, top15.values, color='steelblue')
    axes[2].set_title('Top 15 Feature Importances')
    axes[2].set_xlabel('Importance')
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

train_score = rf.score(X_train, y_train)
test_score  = rf.score(X_test,  y_test)
plot_rf_curves(oob_errors, train_score, test_score)

#Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Random Forest Model: Confusion Matrix for Multiclass attack Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Evaluate
y_pred = rf.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("\nClassification Report:\n",
      classification_report(y_test_labels, y_pred_labels))

plot_confusion_matrix(y_test_labels, y_pred_labels, list(le.classes_))
