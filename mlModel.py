import re
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix

class Model:
    def __init__(self, X, y, model_architecture, vectorizer, random_seed=42, test_size=0.2) -> None:

        # Mask to identify the lines where X_train are not NaN values
        mask = X.notnull()

        # Apply the mask to X and y
        X = X[mask]
        y = y[mask]

        self.X = X
        self.y = y
        self.model_instance = model_architecture
        self.vectorizer = vectorizer
        self.random_seed = random_seed
        self.test_size = test_size

        # the pipeline as defined previously
        self.pipeline = Pipeline([
        ("Vectorizer", self.vectorizer),
        ("Model_Architecture", self.model_instance)
        ])

        # train test split using the above X, y, test_size and random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_seed)

    def preprocess(self, text):
        def remove_twitter_handles_url(text):
            twitter_handle_pattern = r'@[\w_]+'
            url_pattern = r'https?://\S+|www\.\S+'
            
            no_handle = re.sub(twitter_handle_pattern, '', text)
            cleaned_text = re.sub(url_pattern, '', no_handle)

            return cleaned_text
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        cleaned_text = remove_twitter_handles_url(text)
        tokens = word_tokenize(cleaned_text)
        words = [word.lower() for word in tokens if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def fit(self):

        # Preprocess X_train to handle NaN values
        self.X_train = self.X_train.fillna('')  

        # fit self.pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)

        # fit self.pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)

    def fit_with_grid_search(self, parameters):

        # Preprocess X_train to handle NaN values
        self.X_train = self.X_train.fillna('')  

        # Use GridSearchCV to find the best parameters
        self.grid_search = GridSearchCV(self.pipeline, parameters, cv=5, n_jobs=-1)
        self.grid_search.fit(self.X_train, self.y_train)

    def predict(self):
        return self.pipeline.predict(self.X_test)

    def predict_proba(self):
        return self.pipeline.predict_proba(self.X_test)

    def class_report(self, class_labels):
        # the report function as defined previously
        print(classification_report(self.y_test, self.predict(), target_names=class_labels))
        confusion_matrix1 = confusion_matrix(self.y_test, self.predict())
        fig = px.imshow(
            confusion_matrix1, 
            text_auto=True, 
            title="Confusion Matrix", width=1000, height=800,
            labels=dict(x="Predicted", y="True Label"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale='Blues'
        )
        fig.show()