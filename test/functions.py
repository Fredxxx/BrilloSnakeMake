import json
import os


def readJSON(jsonPath):
    if os.path.exists(jsonPath):
        try:
            with open(jsonPath, 'r', encoding='utf-8') as datei:
                j = json.load(datei)
                return j
        except KeyError as e:
            print(f"Error open JSON: {e}")
            return "error JSON read"
    else:
        print(f"Error: '{jsonPath}' could not be found.")
    return "error JSON read"
