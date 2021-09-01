# pip install pandas seaborn wordcloud nltk sklearn scikitplot matplotlib re
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import datetime




df_train = pd.read_csv("train2.txt", delimiter=';', names=['text', 'label'])
df_val = pd.read_csv("val2.txt", delimiter=';', names=['text', 'label'])

df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)
print("Shape of the DataFrame:", df.shape)
df.sample(5)
print("Defining custom_encoder function")
def custom_encoder(df):
    df.replace(to_replace ="surprise", value =1, inplace=True)
    df.replace(to_replace ="love", value =1, inplace=True)
    df.replace(to_replace ="joy", value =1, inplace=True)
    df.replace(to_replace ="fear", value =0, inplace=True)
    df.replace(to_replace ="anger", value =0, inplace=True)
    df.replace(to_replace ="sadness", value =0, inplace=True)
print("Calling custom_encoder")
custom_encoder(df['label'])

# object of WordNetLemmatizer
lm = WordNetLemmatizer()

print("Definind text_transformation function")
def text_transformation(df_col):
    corpus = []
    print("coprus empty")
    for item in df_col:
        print()
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(
            stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
        print(datetime.datetime.now(),"appending new item to copus")
    print("Done transforming")
    return corpus

print("Transforming trainning text")
corpus = text_transformation(df['text'])
print("definind rcParams")
mpl.rcParams['figure.figsize'] = 20, 8
word_cloud = ""
for row in corpus:
    for word in row:
        word_cloud += " ".join(word)
wordcloud = WordCloud(width=1000, height=500, background_color='white',
                      min_font_size=10).generate(word_cloud)
print("Showing wordClout")
#plt.imshow(wordcloud)

print("Count Vectorizer")
cv = CountVectorizer(ngram_range=(1, 2))
print("Fitting transform")
traindata = cv.fit_transform(corpus)
print("defininf X = trainData")
X = traindata
print("defininf y = label")
y = df.label


# Model
print("Defining Model")
parameters = {'max_features': ('auto', 'sqrt'),
              'n_estimators': [500, 1000, 1500],
              'max_depth': [5, 10, None],
              'min_samples_split': [5, 10, 15],
              'min_samples_leaf': [1, 2, 5, 10],
              'bootstrap': [True, False]}
print("Model Parameters: ",parameters)
grid_search = GridSearchCV(RandomForestClassifier(
), parameters, cv=5, return_train_score=True, n_jobs=-1)
print("Grid search done")
print("Fitting x and y")
grid_search.fit(X, y)
print("Best params")
grid_search.best_params_
print("Gonna loop 432 times")
for i in range(432):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test Score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])
print("Setting Classifier")
rfc = RandomForestClassifier(max_features=grid_search.best_params_['max_features'],
                             max_depth=grid_search.best_params_['max_depth'],
                             n_estimators=grid_search.best_params_[
                                 'n_estimators'],
                             min_samples_split=grid_search.best_params_[
                                 'min_samples_split'],
                             min_samples_leaf=grid_search.best_params_[
                                 'min_samples_leaf'],
                             bootstrap=grid_search.best_params_['bootstrap'])
print("Fitting classifier")
rfc.fit(X, y)

print("Running some tests")
print("Test data transform")
# Test data transformation
test_df = pd.read_csv('test.txt', delimiter=';', names=['text', 'label'])
X_test, y_test = test_df.text, test_df.label
# encode the labels into two classes , 0 and 1
test_df = custom_encoder(y_test)
# pre-processing of text
test_corpus = text_transformation(X_test)
# convert text data into vectors
testdata = cv.transform(test_corpus)
# predict the target
print("Running predictions")
predictions = rfc.predict(testdata)
print("Predictions:",predictions)

mpl.rcParams['figure.figsize'] = 10, 5
plot_confusion_matrix(y_test, predictions)
acc_score = accuracy_score(y_test, predictions)
pre_score = precision_score(y_test, predictions)
rec_score = recall_score(y_test, predictions)
print('Accuracy_score: ', acc_score)
print('Precision_score: ', pre_score)
print('Recall_score: ', rec_score)
print("-"*50)
cr = classification_report(y_test, predictions)
print(cr)

predictions_probability = rfc.predict_proba(testdata)
print("Predictions probability",predictions_probability)
fpr, tpr, thresholds = roc_curve(y_test, predictions_probability[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.show()

print("Defininf checked fucntion")
# For testing
def expression_check(prediction_input):
    if prediction_input == 0:
        print("Input statement has Negative Sentiment.")
    elif prediction_input == 1:
        print("Input statement has Positive Sentiment.")
    else:
        print("Invalid Statement.")
# function to take the input statement and perform the same transformations we did earlier

print("Defining sentiment prefictor")
def sentiment_predictor(input):
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = rfc.predict(transformed_input)
    expression_check(prediction)

print("Going to test twice")
input1 = ["Sometimes I just want to punch someone in the face."]
input2 = ["I bought a new phone and it's so good."]

sentiment_predictor(input1)
sentiment_predictor(input2)
print("Finished")
