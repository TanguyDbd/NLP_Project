from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in tokens if word.isalnum()]
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization using WordNetLemmatizer
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)


def matrix_report(rep_y_true, rep_y_pred, class_labels):
    # print a classification report of the predictions
    print(classification_report(rep_y_true, rep_y_pred, target_names=class_labels))

    # styling the confusion matrix
    confusion_matrix_kwargs = dict(
    text_auto=True, 
    title="Confusion Matrix", width=1000, height=800,
    labels=dict(x="Predicted", y="True Label"),
    x=class_labels,
    y=class_labels,
    color_continuous_scale='Blues')

    # create a confusion matrix and pass it to imshow to visualize it
    confusion_matrix1 = confusion_matrix(rep_y_true, rep_y_pred)
    fig = px.imshow(
        confusion_matrix1, 
        **confusion_matrix_kwargs
        )
    fig.show()