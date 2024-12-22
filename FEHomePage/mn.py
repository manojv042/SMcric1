import requests
import random
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import csv

app = Flask(__name__)
CORS(app)

# Function to read data from the CSV file
def read_csv_data(file_path):
    team1 = set()
    team2 = set()
    cities = set()

    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                team1_name = row.get('team1')  
                team2_name = row.get('team2')  
                city_name = row.get('city')  

                if team1_name:
                    team1.add(team1_name)
                if team2_name:
                    team2.add(team2_name)
                if city_name:
                    cities.add(city_name)
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], [], []

    return list(team1), list(team2), list(cities)

# RapidAPI Configuration
headers = {
    'x-rapidapi-key': "YOUR_RAPIDAPI_KEY",
    'x-rapidapi-host': "cricbuzz-cricket.p.rapidapi.com"
}

# Route to fetch dropdown data for teams and cities
@app.route('/dropdown_data', methods=['GET'])
def dropdown_data():
    try:
        csv_file_path = r'C:\Users\manoj\SMcric\FEHomePage\output2.csv'

        # Read team1, team2, and cities from the CSV file
        team1, team2, cities = read_csv_data(csv_file_path)

        # Prepare the data in the expected format
        data = {
            "team1": [{"name": team, "icon": "🏏"} for team in team1],
            "team2": [{"name": team, "icon": "🏏"} for team in team2],
            "cities": cities
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": "Failed to fetch dropdown data", "message": str(e)}), 500

# Route to predict match outcome based on user input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the frontend (in JSON format)
        data = request.json
        
        # Example of expected data format from frontend
        team1 = data.get('team1')
        team2 = data.get('team2')
        city = data.get('city')
        required_runs = data.get('required_runs')
        required_overs = data.get('required_overs')
        required_wickets = data.get('required_wickets')
        toss_winner = data.get('toss_winner')
        toss_decision = data.get('toss_decision')
        target_runs = data.get('target_runs', None)

        # Simulating prediction (replace this with your actual prediction logic)
        team1_win_probability = random.uniform(40, 60)  # Mock value for demo
        team2_win_probability = 100 - team1_win_probability  # The complement
        
        # Prepare the response
        response = {
            "team1_win_probability": team1_win_probability,
            "team2_win_probability": team2_win_probability,
            "match_details": {
                "team1": team1,
                "team2": team2,
                "city": city,
                "required_runs": required_runs,
                "required_overs": required_overs,
                "required_wickets": required_wickets,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "target_runs": target_runs
            }
        }

        # Return the response in JSON format
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500

# Route for live matches (optional, not used directly in the frontend, but you can add it if needed)
@app.route('/live_matches')
def live_matches():
    url = "https://cricbuzz-cricket.p.rapidapi.com/matches/v1/recent"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return jsonify(data)
        else:
            return jsonify({"error": "Failed to fetch live matches", "status_code": response.status_code}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "An error occurred while fetching live matches", "message": str(e)}), 500

# Route to display the homepage
@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
