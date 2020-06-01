from typing import List, Tuple, Optional
import json
from pathlib import Path
import re
import copy
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression


def load_json(path: str = "data", filename: str = "json_table.json") -> Tuple[dict, str]:
    with open(Path(path) / filename, 'r', encoding='UTF-8') as f:
        file_as_json = json.load(f)
    with open(Path(path) / filename, 'r', encoding='UTF-8') as f:
        file_as_str = f.read()
    return file_as_json, file_as_str


def convert_json_to_pd(input_data: Tuple[dict, str]) -> pd.DataFrame():

    def extract_teams_from_json(input_json: dict) -> List[str]:
        return [
            _['div']['div']['#text']
            for _ in input_json['tbody']['tr'][0]['th'][1:]
        ]

    def extract_match_results_from_text(input_text: str) -> List[str]:
        return re.findall(pattern="\"[0-2]:[0-2]\"", string=input_text)

    def join_teams_and_results(teams: List[str], results: List[str]) -> List[List[str]]:
        output: List = []
        t = len(teams)
        for r in range(len(results)):
            team1 = teams[r // (t - 1)]
            team2 = [_t for _t in teams if _t != team1][r % (t - 1)]
            if team1.lower() < team2.lower():
                result = results[r]
                output.append([team1, team2, result])
        return output

    teams = extract_teams_from_json(input_data[0])
    results = [
        re.findall(pattern="[0-2]:[0-2]", string=_)[0]
        for _ in extract_match_results_from_text(input_data[1])
    ]

    if len(results) != len(teams) * (len(teams) - 1):
        raise RuntimeError(f"{len(teams)} teams, but {len(results)} results")

    return pd.DataFrame.from_records(join_teams_and_results(teams, results), columns=['team1', 'team2', 'result'])


def convert_input_data_to_regression_data(pdf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    teams = list(set(list(pdf['team1']) + list(pdf['team2'])))

    train = pd.DataFrame(columns=['outcome'] + teams)
    test = pd.DataFrame(columns=['outcome'] + teams)

    def convert_result_to_points(result: str) -> Optional[int]:
        if result == '0:0':
            return None
        if result == '2:0':
            return 3
        if result == '2:1':
            return 2
        if result == '1:2':
            return 1
        if result == '0:2':
            return 0
        raise ValueError(f"Invalid result {result}")

    for index, row in pdf.iterrows():
        outcome = convert_result_to_points(row['result'])
        row_dict = {
            'outcome': outcome,
            'team1': row['team1'],
            'team2': row['team2']
        }
        row_dict.update(
            {
                _t: 1 if row['team1'] == _t else -1 if row['team2'] == _t else 0
                for _t in teams
            }
        )
        if outcome is not None:
            train = train.append(row_dict, ignore_index=True)
        else:
            test = test.append(row_dict, ignore_index=True)

    # removing teams without matches
    colsums = train.drop(['outcome', 'team1', 'team2'], axis=1).abs().sum()
    teams_out = [
        _t
        for _t, _matches
        in colsums.iteritems()
        if _matches == 0
    ]
    teams_in = [_t for _t in teams if _t not in teams_out]

    train_filter = (
        train[(train['team1'].isin(teams_in)) & (train['team2'].isin(teams_in))]
            .drop(columns=['team1', 'team2'] + teams_out)
    )
    cols = [_c for _c in train_filter.columns if _c != 'outcome']
    cols.sort()
    train_filter = train_filter[['outcome'] + cols]
    test_filter = (
        test[(test['team1'].isin(teams_in)) & (test['team2'].isin(teams_in))]
            .drop(columns=['outcome', 'team1', 'team2'] + teams_out)
    )
    cols = list(test_filter.columns)
    cols.sort()
    test_filter = test_filter[cols]

    # train_filter.to_csv('train.csv', index=False)
    # test_filter.to_csv('test.csv', index=False)

    return train_filter, test_filter, teams_out


def convert_to_compact_format(all_data: pd.DataFrame) -> pd.DataFrame:

    teams = list(set(all_data.columns) - {'outcome'})

    output = pd.DataFrame(columns=['team1', 'team2', 'outcome'])
    for index, row in all_data.iterrows():
        team1 = None
        team2 = None
        for team in teams:
            if row[team] == 1:
                team1 = team
            if row[team] == -1:
                team2 = team
        if team1 is None or team2 is None:
            raise ValueError(f"Cannot convert row with index {index} to predicted_data")
        row_dict = {
            'team1': team1,
            'team2': team2,
            'outcome': row['outcome']
        }
        output = output.append(row_dict, ignore_index=True)

    return output


def fit_regression_model(regression_data: pd.DataFrame) -> LinearRegression:
    # TODO: should be regularized to outcome >= 0 and <= 3
    model = LinearRegression()
    if regression_data.columns[0] != 'outcome':
        raise NameError("First column of regression data must be named outcome")
    model.fit(
        regression_data[regression_data.columns[1:]].values.tolist(),
        regression_data[regression_data.columns[0]].values.tolist()
    )
    return model


def predict_with_regression_model(regression_model: LinearRegression, regression_data: pd.DataFrame) -> pd.DataFrame:
    predictions_vector = regression_model.predict(regression_data)
    predictions = copy.deepcopy(regression_data)
    predictions['outcome'] = predictions_vector
    return predictions


def calculate_points(all_data_compact_format: pd.DataFrame) -> pd.DataFrame:

    teams = {}
    for index, row in all_data_compact_format.iterrows():
        if row['team1'] not in teams.keys():
            teams[row['team1']] = 0
        if row['team2'] not in teams.keys():
            teams[row['team2']] = 0
        teams[row['team1']] += row['outcome']
        teams[row['team2']] += (3 - row['outcome'])

    return (
        pd.DataFrame.from_records(
            [[_k, teams[_k]] for _k in teams.keys()], columns=['Team', 'Points']
        )
        .sort_values(by='Points', ascending=False)
        .reset_index().drop(columns=['index'])
    )


def convert_points_to_outcome(points: pd.DataFrame, teams_out: List[str]) -> pd.DataFrame:
    outcome = copy.deepcopy(points)
    teams_count = points.shape[0]
    maximal_point_count = 3 * teams_count
    outcome['Percentage'] = 100 * outcome['Points'] / maximal_point_count

    # adding points for the teams that were not part of the analysis
    teams_out_count = len(teams_out)
    teams_out_points = 3 * teams_out_count
    outcome['Points'] += (teams_out_points * outcome['Percentage']) / 100

    outcome['Order'] = range(1, teams_count + 1)

    return outcome[['Order', 'Team', 'Points', 'Percentage']]


if __name__ == '__main__':
    input_data = load_json()
    input_data_pd = convert_json_to_pd(input_data)
    train_data, test_data, teams_out = convert_input_data_to_regression_data(input_data_pd)
    print(f"No data for teams {', '.join(teams_out)}. Fitting regression on other teams.")
    regression_model = fit_regression_model(train_data)
    predictions = predict_with_regression_model(regression_model, test_data)
    all_data = train_data.append(predictions)
    all_data_compact_format = convert_to_compact_format(all_data)
    points = calculate_points(all_data_compact_format)
    outcome = convert_points_to_outcome(points, teams_out)
    outcome.to_csv('outcome.csv', index=False)
