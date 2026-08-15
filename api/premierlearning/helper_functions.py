import json

def read_json(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except:
        pass