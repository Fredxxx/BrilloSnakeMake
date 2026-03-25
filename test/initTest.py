import fucntions as f
import os
import sys

# Der absolute Pfad des Ordners, in dem das Skript liegt
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# Erstellen Sie den vollständigen Pfad zur Zieldatei
file_path = os.path.join(script_dir, 'daten.txt')

jsonPath = 'C:\\Users\\Fred\\Documents\\GitHub\\BrilloSnakeMake\\brilloPara.txt'


f.readJSON(jsonPath)