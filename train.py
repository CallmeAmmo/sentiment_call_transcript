# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from collections import Counter

# # Sample dataset with extracted phrases and sentiment labels
# data = [
#     {"transcript_id": "T1", "extracted_phrases": "profit increase revenue growth", "label": "positive"},
#     {"transcript_id": "T2", "extracted_phrases": "loss decline drop warning", "label": "negative"},
#     {"transcript_id": "T3", "extracted_phrases": "profit warning mixed performance", "label": "negative"},
#     {"transcript_id": "T4", "extracted_phrases": "strong earnings revenue up", "label": "positive"},
#     {"transcript_id": "T5", "extracted_phrases": "decline slowdown loss", "label": "negative"},
#     {"transcript_id": "T6", "extracted_phrases": "growth expansion profit increase", "label": "positive"},
# ]

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Step 1: Split into Train and Test Data (80% - 20%)
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# # Step 2: Extract Positive and Negative Phrases for Vocabulary (from TRAINING data only)
# positive_phrases = []
# negative_phrases = []

# for _, row in train_df.iterrows():
#     phrases = row['extracted_phrases'].split()  # Tokenize extracted phrases
#     if row['label'] == 'positive':
#         positive_phrases.extend(phrases)
#     elif row['label'] == 'negative':
#         negative_phrases.extend(phrases)

# # Step 3: Create Vocabulary using CountVectorizer
# # Combine positive and negative phrases for vocabulary
# all_phrases = positive_phrases + negative_phrases
# vectorizer = CountVectorizer(vocabulary=all_phrases)

# # Step 4: Generate Features using CountVectorizer for both train and test data
# X_train = vectorizer.transform(train_df['extracted_phrases'])
# y_train = train_df['label']

# X_test = vectorizer.transform(test_df['extracted_phrases']) 
# y_test = test_df['label']

# # Step 5: Train a Logistic Regression Classifier
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Step 6: Evaluate the Model on Test Data
# y_pred = model.predict(X_test)

# # Step 7: Classification Report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Optional: Print the top phrases from the positive and negative vocabularies
# print("\nTop Positive Phrases:", Counter(positive_phrases).most_common(5))
# print("Top Negative Phrases:", Counter(negative_phrases).most_common(5))


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

# Sample dataset with extracted phrases and sentiment labels
data = [
    {"transcript_id": "T1", "extracted_phrases": "profit increase revenue growth", "label": "positive"},
    {"transcript_id": "T2", "extracted_phrases": "loss decline drop warning", "label": "negative"},
    {"transcript_id": "T3", "extracted_phrases": "profit warning mixed performance", "label": "negative"},
    {"transcript_id": "T4", "extracted_phrases": "strong earnings revenue up", "label": "positive"},
    {"transcript_id": "T5", "extracted_phrases": "decline slowdown loss", "label": "negative"},
    {"transcript_id": "T6", "extracted_phrases": "growth expansion profit increase", "label": "positive"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Initialize variables for storing results
kf = KFold(n_splits=5, shuffle=True, random_state=42)
overall_metrics = []

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    print(f"Fold {fold + 1}")
    
    # Split the data into train and test sets for this fold
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Extract Positive and Negative Phrases for Vocabulary (from TRAINING data only)
    positive_phrases = set()
    negative_phrases = set()

    for _, row in train_df.iterrows():
        phrases = row['extracted_phrases'].split()  # Tokenize extracted phrases
        if row['label'] == 'positive':
            positive_phrases.update(phrases)
        elif row['label'] == 'negative':
            negative_phrases.update(phrases)

    # Combine Positive and Negative Phrases into Vocabulary
    all_phrases = list(positive_phrases.union(negative_phrases))  # Ensure unique terms only

    # Create CountVectorizer using the Vocabulary
    vectorizer = CountVectorizer(vocabulary=all_phrases)

    # Generate Features for Train and Test Data
    X_train = vectorizer.transform(train_df['extracted_phrases'])
    y_train = train_df['label']

    X_test = vectorizer.transform(test_df['extracted_phrases'])
    y_test = test_df['label']

    # Train a Logistic Regression Classifier
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and Evaluate the Model
    y_pred = model.predict(X_test)

    # Calculate metrics for this fold
    report = classification_report(y_test, y_pred, output_dict=True)
    overall_metrics.append(report)

    # Print metrics for the current fold
    print(f"Classification Report for Fold {fold + 1}:")
    print(classification_report(y_test, y_pred))

# Average the metrics across all folds
average_metrics = {
    metric: sum(fold_metrics[metric]['f1-score'] for fold_metrics in overall_metrics if metric in fold_metrics) / kf.n_splits
    for metric in ['positive', 'negative']
}

print("\nAverage Metrics Across All Folds:")
for sentiment, score in average_metrics.items():
    print(f"{sentiment.capitalize()} F1-Score: {score:.2f}")
