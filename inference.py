import pickle
import pandas as pd


model = pickle.load(open('covid_dataset.pkl','rb'))
Breathing_Problem_encoder = pickle.load(open('Breathing_Problem_encoder.pkl', 'rb'))
Fever_encoder = pickle.load(open('Fever_encoder.pkl', 'rb'))
Dry_Cough_encoder = pickle.load(open('Dry_Cough_encoder.pkl', 'rb'))
Sore_throat_encoder = pickle.load(open('Sore_throat_encoder.pkl', 'rb'))
Hyper_Tension_encoder = pickle.load(open('Hyper_Tension_encoder.pkl', 'rb'))
Fatigue_encoder = pickle.load(open('Fatigue_encoder.pkl', 'rb'))
Abroad_travel_encoder = pickle.load(open('Abroad_travel_encoder.pkl', 'rb'))
Contact_with_COVID_Patient_encoder = pickle.load(open('Contact_with_COVID_Patient_encoder.pkl', 'rb'))
Attended_Large_Gathering_encoder = pickle.load(open('Attended_Large_Gathering_encoder.pkl', 'rb'))
Visited_Public_Exposed_Places_encoder = pickle.load(open('Visited_Public_Exposed_Places_encoder.pkl', 'rb'))
Family_working_in_Public_Exposed_Places_encoder = pickle.load(open('Family_working_in_Public_Exposed_Places_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
class_names = [0, 1]  # int values





def predict(df):
    df = df[['Breathing Problem','Fever','Dry Cough','Sore throat','Hyper Tension','Fatigue ','Abroad travel','Contact with COVID Patient','Attended Large Gathering','Visited Public Exposed Places','Family working in Public Exposed Places']]
    df['Breathing Problem'] = Breathing_Problem_encoder.transform(df['Breathing Problem'])
    df['Fever'] = Fever_encoder.transform(df['Fever'])
    df['Dry Cough'] = Dry_Cough_encoder.transform(df['Dry Cough'])
    df['Sore throat'] = Sore_throat_encoder.transform(df['Sore throat'])
    df['Hyper Tension'] = Hyper_Tension_encoder.transform(df['Hyper Tension'])
    df['Fatigue '] = Fatigue_encoder.transform(df['Fatigue '])
    df['Abroad travel'] = Abroad_travel_encoder.transform(df['Abroad travel'])
    df['Contact with COVID Patient'] = Contact_with_COVID_Patient_encoder.transform(df['Contact with COVID Patient'])
    df['Attended Large Gathering'] = Attended_Large_Gathering_encoder.transform(df['Attended Large Gathering'])
    df['Visited Public Exposed Places'] = Visited_Public_Exposed_Places_encoder.transform(df['Visited Public Exposed Places'])
    df['Family working in Public Exposed Places'] = Family_working_in_Public_Exposed_Places_encoder.transform(df['Family working in Public Exposed Places'])
    df = pd.DataFrame(scaler.transform(df))
    numpy_array = df.to_numpy()
    predictions = model.predict(numpy_array)
    output = [class_names[class_predicted] for class_predicted in predictions]
    return output

