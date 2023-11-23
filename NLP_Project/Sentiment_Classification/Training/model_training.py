import spacy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Split the dataframe into test and train data
def split_data(df):
    df = df[:100]
    df['label_num'] = df['label'].map({'FAKE' : 0, 'REAL': 1})
    print(df.columns)
    df = df.drop(['label','title','Unnamed: 0'], axis = 1)
    y = df['label_num'].values
    
    nlp = spacy.load('en_core_web_lg')
    df['vector'] = df['text'].apply(lambda text: nlp(text).vector)  

    X_train, X_test, y_train, y_test = train_test_split(
        df['vector'].values, y , test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    
    return data


# Train the model, return the model
def train_model(data):
    clf = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
    X_train_2d = np.stack(data["train"]["X"])
    X_test_2d = np.stack(data["test"]["X"])
    clf.fit(X_train_2d, data["train"]["y"])
    
    

# Evaluate the metrics for the model
def get_model_metrics(model, data):
    y_pred = model.predict(X_test_2d)
    accuracy = accuracy_score(data["test"]["y"], y_pred)
    metrics = {"accuracy": accuracy}
    return metrics




def main():
    print("Running train.py")
    
    data_dir = "data"
    data_file = os.path.join(data_dir, 'news.csv')
    train_df = pd.read_csv(data_file)

    data = split_data(train_df)

    # Train the model
    model = train_model(data)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()