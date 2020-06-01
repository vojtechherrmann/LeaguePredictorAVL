from typing import List, Tuple
import json
from pathlib import Path
import re

import pandas as pd


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

    return pd.DataFrame.from_records(join_teams_and_results(teams, results))


if __name__ == '__main__':
    input_data = load_json()
    input_data_pd = convert_json_to_pd(input_data)

    test = 1
