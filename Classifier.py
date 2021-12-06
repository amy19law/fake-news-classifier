# Created by Amy Law

# Import Libraires/Modules
import pandas as pd
import pickle
from IPython.display import FileLink
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('Outputs from Fake News Classifier & Model Stats')

# Data Frame is the Training CSV
dataframe = pd.read_csv('train.csv')

# Convert the 0s in the Labels to True & the 1s to Fake
conversion = {0: 'True',1: 'Fake'}
dataframe['label'] = dataframe['label'].replace(conversion)

# Created by Amy Law

# Makes Sure that the Training Data is Relatively Balanced
dataframe.label.value_counts()

# Trains the Model to find Relationship between Text & Label of True or False given
trainX,testX,trainY,testY = train_test_split(dataframe['text'], dataframe['label'], test_size = 0.8, random_state = 7, shuffle = True)
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df = 0.75)

# Convert Pandas Object into Readable Strings
vecTrain = tfidf_vectorizer.fit_transform(trainX.values.astype('U'))
vecTest = tfidf_vectorizer.transform(testX.values.astype('U'))

# Passive Aggressive Classifier creates a Hyperplane between True or False & then Adjusts Itself depending on the Article & Correct Itself
# Creates Model
PAC = PassiveAggressiveClassifier(max_iter = 50)

# Fitting Model
PAC.fit(vecTrain,trainY)

# Run the Model Against the Given Train Data, originally used to create Model
# Make Prediction on Test Set
predY = PAC.predict(vecTest)
score = accuracy_score(testY,predY)

# Created by Amy Law

# Model Evaluation
print('')
print(f'PAC Accuracy: {round(score*100,2)}%')
print('')
print(f"Classification Report : \n\n{classification_report(testY, predY)}")
print('')

# Display K-Fold Accuracy
k = tfidf_vectorizer.transform(dataframe['text'].values.astype('U'))
results_kfold = cross_val_score(PAC,k,dataframe['label'].values,cv = 5) # "cv" specifies the number of cross-validation splits 
print(f'K-Fold Accuracy: {round(results_kfold.mean()*100,2)}%')

# Reads in More Data not Related to the Test Data & sees how well the Model does
dataframe_true = pd.read_csv('true.csv')
dataframe_true['label'] = 'True'

dataframe_fake = pd.read_csv('fake.csv')
dataframe_fake['label'] = 'Fake'

dataframe_final = pd.concat([dataframe_true,dataframe_fake])
dataframe_final = dataframe_final.drop(['subject','date'], axis = 1)

def findLabel(newtext):
    vec_newtest = tfidf_vectorizer.transform([newtext])
    predY1 = PAC.predict(vec_newtest)
    return predY1[0]

# Created by Amy Law

# Give Label 1 if it Predicts True, 0 if it Predicts False, get the 1s divided by Total
print('')
print('Predicting True Articles')
true_result = (sum([1 if findLabel((dataframe_true['text'][i]))=='True' else 0 for i in range(len(dataframe_true['text']))])/dataframe_true['text'].size)
print(f'True: {round(true_result*100,2)}%')

#Give Label 1 if it Predicts False, 0 if it Predicts True, get the 1s divided by Total
print('')
print('Predicting Fake Articles')
fake_result = (sum([1 if findLabel((dataframe_fake['text'][i]))=='Fake' else 0 for i in range(len(dataframe_fake['text']))])/dataframe_fake['text'].size)
print(f'Fake: {round(fake_result*100,2)}%')

# Save Trained Model
with open('model.model', 'wb') as f:
    pickle.dump(PAC, f)
    
print('')
print('Saving Model')
FileLink(r'model.model')

# Created by Amy Law
