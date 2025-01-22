import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

# Define the folders for positive and negative phrases
positive_folder = "sentiment_files/positive/"
negative_folder = "sentiment_files/negative/"


def get_data():
    # df = pd.read_csv(r'CallEarningTranscripts\CallEarningTranscripts\price_mapped_transcripts_multiple_returns_new.csv')
    df = pd.read_csv(r'CallEarningTranscripts/CallEarningTranscripts/price_mapped_transcripts_multiple_returns_new.csv')

    data = df[['pdf_file','d1_n1_return']]
    data.dropna(inplace=True)

    data['sentiment_class'] = data['d1_n1_return'].apply(lambda x: 'positive' if x > 0 else 'negative')
    return data[['pdf_file', 'sentiment_class']]




# Function to read phrases based on class and file path
def read_phrases_from_folder(pdf_file, sentiment_class):

    file_name = pdf_file.split('_')[1].split('.')[0]

    folder = positive_folder if sentiment_class == "positive" else negative_folder
    file_path = os.path.join(folder, f"{sentiment_class}_phrases_{file_name}_output.txt")
    
    try:
       with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace("\n", " ").strip()  # Read and strip whitespace
    except FileNotFoundError:
        return ""  # Return an empty string if the file is not found


def train_model(df):

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
            if row['sentiment_class'] == 'positive':
                positive_phrases.update(phrases)
            elif row['sentiment_class'] == 'negative':
                negative_phrases.update(phrases)

        # Combine Positive and Negative Phrases into Vocabulary
        all_phrases = list(positive_phrases.union(negative_phrases))  # Ensure unique terms only

        # Create CountVectorizer using Unigrams, Bigrams, and Trigrams
        vectorizer = CountVectorizer(vocabulary=all_phrases, ngram_range=(1,3))

        # Generate Features for Train and Test Data
        X_train = vectorizer.transform(train_df['extracted_phrases'])
        y_train = train_df['sentiment_class']

        X_test = vectorizer.transform(test_df['extracted_phrases'])
        y_test = test_df['sentiment_class']

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



if __name__ == "__main__":
    df  = get_data()

    # Add a column for extracted phrases by reading the text files dynamically
    df["extracted_phrases"] = df.apply(
        lambda row: read_phrases_from_folder(row["pdf_file"], row["sentiment_class"]), axis=1
    )

    train_model(df)

