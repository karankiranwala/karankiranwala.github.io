# Import necessary libraries

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from flask import Flask, render_template, request

# Load the dataset
data = pd.read_csv(r'C:\Users\Karan\Desktop\Real Estate Price Prediction\406.csv')  # Replace 'your_dataset.csv' with the actual filename

# Define the features (independent variables) and the target (dependent variable)
features = data[['Transaction Size (sq.m)', 'lat', 'lon', 'Metro_Dist', 'Mall_Dist', 'Landmark_Dist', 'Room(s)', 'Registration type_Ready', 'Is Free Hold?_Non Free Hold', 'Parking']]
target = data['Amount']

# Encode the 'Area' column using LabelEncoder
area_encoder = LabelEncoder()
features['Area'] = area_encoder.fit_transform(data['Area'])

# Determine the embedding size based on the number of unique areas
embedding_size = min(53, len(data['Area'].unique()) // 2)

# Create the embedding model
embedding_model = Sequential()
embedding_model.add(Embedding(len(data['Area'].unique()), embedding_size, input_length=1))
embedding_model.add(Dense(1, activation='relu'))
embedding_model.compile(loss='mse', optimizer='adam')

# Fit the embedding model
embedding_model.fit(features['Area'], target, epochs=10, verbose=0)

# Create the random forest regression model
model = RandomForestRegressor()

# Fit the model
model.fit(features, target)

# Create Flask app
app = Flask(__name__, template_folder='templates')

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the user input
        transaction_size = float(request.form['transaction_size'])
        area = request.form['area']
        rooms = int(request.form['rooms'])
        registration_ready = int(request.form['registration_ready'])
        free_hold = int(request.form['free_hold'])
        parking = int(request.form['parking'])
        
        # Encode the selected area using the label encoder
        area_encoded = area_encoder.transform([area])[0]
        
        # Use the embedding model to get the embedded representation
        area_embedding = embedding_model.predict(pd.DataFrame({'Area': [area_encoded]}))
        
        # Create a prediction input with all the features
        prediction_input = [[transaction_size, area_embedding[0][0], features['lat'][0], features['lon'][0], features['Metro_Dist'][0], features['Mall_Dist'][0], features['Landmark_Dist'][0], rooms, registration_ready, free_hold, parking]]
        
        # Use the random forest model to make predictions
        prediction = model.predict(prediction_input)
        
        # Return the prediction to the user
        return render_template('index.html', areas=data['Area'].unique(), prediction=prediction[0])
    
    # If it's a GET request, render the initial HTML page
    return render_template('index.html', areas=data['Area'].unique())

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)