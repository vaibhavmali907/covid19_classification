import pandas as pd
import pickle


# data visualization library
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Model evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'

dataset = pd.read_csv('Covid Dataset.csv')
dataset.head()

dataset.describe(include='all')

print(dataset.shape)

print(dataset.nunique())

print(dataset[dataset['COVID-19'] == 'Yes'].count())

print(dataset[dataset['COVID-19'] == 'No'].count())

dataset.isnull().sum()

dataset = dataset.drop('Running Nose',axis=1)
dataset = dataset.drop('Chronic Lung Disease',axis=1)
dataset = dataset.drop('Headache',axis=1)
dataset = dataset.drop('Heart Disease',axis=1)
dataset = dataset.drop('Diabetes',axis=1)
dataset = dataset.drop('Gastrointestinal ',axis=1)
dataset = dataset.drop('Wearing Masks',axis=1)
dataset = dataset.drop('Sanitization from Market',axis=1)
dataset = dataset.drop('Asthma',axis=1)

#Encoding
from sklearn.preprocessing import LabelEncoder

Breathing_Problem_encoder=LabelEncoder()
dataset['Breathing Problem'] = Breathing_Problem_encoder.fit_transform(dataset['Breathing Problem'])
pickle.dump(Breathing_Problem_encoder, open('Breathing_Problem_encoder.pkl','wb'))

Fever_encoder=LabelEncoder()
dataset['Fever'] = Fever_encoder.fit_transform(dataset['Fever'])
pickle.dump(Fever_encoder, open('Fever_encoder.pkl','wb'))

Dry_Cough_encoder=LabelEncoder()
dataset['Dry Cough'] = Dry_Cough_encoder.fit_transform(dataset['Dry Cough'])
pickle.dump(Dry_Cough_encoder, open('Dry_Cough_encoder.pkl','wb'))

Sore_throat_encoder=LabelEncoder()
dataset['Sore throat'] = Sore_throat_encoder.fit_transform(dataset['Sore throat'])
pickle.dump(Sore_throat_encoder, open('Sore_throat_encoder.pkl','wb'))

Hyper_Tension_encoder=LabelEncoder()
dataset['Hyper Tension'] = Hyper_Tension_encoder.fit_transform(dataset['Hyper Tension'])
pickle.dump(Hyper_Tension_encoder, open('Hyper_Tension_encoder.pkl','wb'))

Fatigue_encoder=LabelEncoder()
dataset['Fatigue '] = Fatigue_encoder.fit_transform(dataset['Fatigue '])
pickle.dump(Fatigue_encoder, open('Fatigue_encoder.pkl','wb'))

Abroad_travel_encoder=LabelEncoder()
dataset['Abroad travel'] = Abroad_travel_encoder.fit_transform(dataset['Abroad travel'])
pickle.dump(Abroad_travel_encoder, open('Abroad_travel_encoder.pkl','wb'))

Contact_with_COVID_Patient_encoder=LabelEncoder()
dataset['Contact with COVID Patient'] = Contact_with_COVID_Patient_encoder.fit_transform(dataset['Contact with COVID Patient'])
pickle.dump(Contact_with_COVID_Patient_encoder, open('Contact_with_COVID_Patient_encoder.pkl','wb'))

Attended_Large_Gathering_encoder=LabelEncoder()
dataset['Attended Large Gathering'] = Attended_Large_Gathering_encoder.fit_transform(dataset['Attended Large Gathering'])
pickle.dump(Attended_Large_Gathering_encoder, open('Attended_Large_Gathering_encoder.pkl','wb'))

Visited_Public_Exposed_Places_encoder=LabelEncoder()
dataset['Visited Public Exposed Places'] = Visited_Public_Exposed_Places_encoder.fit_transform(dataset['Visited Public Exposed Places'])
pickle.dump(Visited_Public_Exposed_Places_encoder, open('Visited_Public_Exposed_Places_encoder.pkl','wb'))

Family_working_in_Public_Exposed_Places_encoder=LabelEncoder()
dataset['Family working in Public Exposed Places'] = Family_working_in_Public_Exposed_Places_encoder.fit_transform(dataset['Family working in Public Exposed Places'])
pickle.dump(Family_working_in_Public_Exposed_Places_encoder, open('Family_working_in_Public_Exposed_Places_encoder.pkl','wb'))

COVID_19_encoder=LabelEncoder()
dataset['COVID-19'] = COVID_19_encoder.fit_transform(dataset['COVID-19'])
pickle.dump(COVID_19_encoder, open('COVID_19_encoder.pkl','wb'))

dataset.head()

dataset.info()

print(dataset[dataset['COVID-19'] == 1].count())

print(dataset[dataset['COVID-19'] == 0].count())

dataset = dataset.astype('category')
dataset.info()

X = dataset.iloc[:, :11]
y = dataset['COVID-19']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test =  sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
l1 = LogisticRegression()
l1.fit(X_train,y_train)

l1_predict = l1.predict(X_test)

cm = confusion_matrix(y_test, l1_predict)
print(cm)

print(classification_report(y_test,l1_predict))

acc_logreg = accuracy_score(y_test, l1_predict)


from sklearn.tree import DecisionTreeClassifier
d1 = DecisionTreeClassifier()
d1.fit(X_train,y_train)

d1_predict = d1.predict(X_test)

cm = confusion_matrix(y_test, d1_predict)
print(cm)

print(classification_report(y_test,d1_predict))

acc_decision_tree = accuracy_score(y_test, d1_predict)


from sklearn.ensemble import RandomForestClassifier
r1 = RandomForestClassifier()
r1.fit(X_train,y_train)

r1_predict = r1.predict(X_test)

cm = confusion_matrix(y_test, r1_predict)
print(cm)

print(classification_report(y_test,r1_predict))

acc_random_forest = accuracy_score(y_test, r1_predict)

from sklearn.neighbors import KNeighborsClassifier
k1 = KNeighborsClassifier(n_neighbors=10)
k1.fit(X_train,y_train)

k1_predict = k1.predict(X_test)

cm = confusion_matrix(y_test,k1_predict)
print(cm)

print(classification_report(y_test,k1_predict))

acc_knn = accuracy_score(y_test, k1_predict)



from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

covid_model = model.predict(X_test)

cm = confusion_matrix(y_test, covid_model)
print(cm)

print(classification_report(y_test,covid_model))

acc_svc = accuracy_score(y_test, covid_model)



models = pd.DataFrame({
    'Model': ['Logistic Regression','Decision Tree','Random Forest','KNN','Support Vector Machines'],
    'Score': [acc_logreg, acc_decision_tree, acc_random_forest, acc_knn , acc_svc]})
models.sort_values(by='Score', ascending=False)

plt.rcParams['figure.figsize']=15,6
sns.set_style("darkgrid")
ax = sns.barplot(x=models.Model, y=models.Score, palette = "rocket", saturation =1.5)
plt.xlabel("Classifier Models", fontsize = 20 )
plt.ylabel("% of Accuracy", fontsize = 20)
plt.title("Accuracy of different Classifier Models", fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


import pickle

pickle.dump(model,open("covid_dataset.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))

from inference import predict

df = pd.read_csv('test.csv')

output = predict(df)

print(output)
