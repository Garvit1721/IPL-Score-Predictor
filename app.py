from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

file_path = r"D:\New folder\pipe.pkl"
file_path1 = r"D:\New folder\data.pkl"

pipe = pickle.load(open(file_path, 'rb'))
data = pickle.load(open(file_path1, 'rb'))

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def welcome():
    result = None
    if request.method == 'POST':
        Team_1 = request.form['Team_1']
        Team_2 = request.form['Team_2']
        ball_left_team1 = request.form['ball_left_team1']
        ball_left_team2 = request.form['ball_left_team2']
        run_by_team1 = request.form['run_by_team1']
        run_by_team2 = request.form['run_by_team2']
        run_left = request.form['run_left']
        wickets_team1 = request.form['wickets_team1']
        wickets_team2 = request.form['wickets_team2']
        crr = request.form['current_run_rate']
        rrr = request.form['required_run_rate']
        city = request.form['city']
        venue = request.form['venue']

        win_Team1 = data[data['Team2_x'] == Team_1].iloc[0].iloc[1]
        win_Team2 = data[data['Team2_x'] == Team_2].iloc[0].iloc[1]

        data_pred = pd.DataFrame({
            'Team1_x': [Team_1],
            'Team2_x': [Team_2],
            'ball_left_with_team1': [ball_left_team1],
            'ball_left_with_team2': [ball_left_team2],
            'run_by_team1': [run_by_team1],
            'run_by_team2': [run_by_team2],
            'runs_left': [run_left],
            'wickets_team1': [wickets_team1],
            'wickets_team2': [wickets_team2],
            'current_run_rate': [crr],
            'required_run_rate': [rrr],
            'city': [city],
            'venue': [venue],
            'win_per_x': [win_Team1],
            'win_per_y': [win_Team2]
        })

        result_proba = pipe.predict_proba(data_pred)
        win_per_Team1 = np.round(result_proba.T[1] * 100, 1)
        win_per_Team2 = np.round(result_proba.T[0] * 100, 1)

        result = {
            'win_per_Team1': win_per_Team1[0],
            'win_per_Team2': win_per_Team2[0]
        }
        print(result)

    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
