import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cca_mel_results=Path(f'{os.pardir}/logs/librispeech_wav2vec_small/cca_mel_dev-clean_sample1.json')
parent_directory = Path('/home/eduardo/PycharmProjects/layerwise-analysis/logs/libriTTS_wav2vec_small/')
cca_word_results= parent_directory / 'cca_word_dev-clean_og.lst'
cca_boundary_results= parent_directory / 'cca_boundary_dev-clean.lst'
cca_prosody_results= parent_directory / 'cca_prosody_dev-clean.lst'


with open(cca_mel_results, 'r') as file:
    cca_mel_results_data = json.load(file)

print(cca_mel_results_data)
fig, ax = plt.subplots()
ax.set_title('CCA between mel spectogram features and \n wav2vec frame level representations')
ax.plot(cca_mel_results_data.keys(), cca_mel_results_data.values())
ax.set_xlabel('wav2vec Layer')
ax.set_ylabel('CCA score')
plt.show()

# with open(cca_word_results, 'r') as file:
#     cca_word_results_data = json.load(file)
# print(cca_word_results_data)

cca_word_results_data = pd.read_csv(cca_word_results, header=None, names=['layer', 'middle', 'score'])


fig, ax = plt.subplots()
ax.plot(cca_word_results_data['layer'], cca_word_results_data['score'])
ax.set_title('CCA between word level representations and word label')
ax.set_xlabel('wav2vec Layer')
ax.set_ylabel('CCA score')
ax.set_xticks(np.arange(12))
plt.show()

cca_prosody_results_data = pd.read_csv(cca_prosody_results, header=None, names=['layer', 'middle', 'score'])

fig, ax = plt.subplots()
ax.plot(cca_prosody_results_data['layer'], cca_prosody_results_data['score'])
ax.set_title('CCA between word level representations and prosodic prominence')
ax.set_xlabel('wav2vec Layer')
ax.set_ylabel('CCA score')
ax.set_xticks(np.arange(12))
plt.show()



cca_boundary_results_data = pd.read_csv(cca_boundary_results, header=None, names=['layer', 'middle', 'score'])

fig, ax = plt.subplots()
ax.plot(cca_boundary_results_data['layer'], cca_boundary_results_data['score'])
ax.set_title('CCA between word level representations and phrase boundary strength')
ax.set_xlabel('wav2vec Layer')
ax.set_ylabel('CCA score')
ax.set_xticks(np.arange(12))
plt.show()



import csv
import numpy as np

mi_results_path = Path(f"{os.pardir}/logs/librispeech_wav2vec_small/mi_word_dev-clean_train-clean-100.lst")
layer_num = []
mi_scores = []

# Read the .lst file as a CSV
with open(mi_results_path, 'r') as file:
    csv_reader = csv.reader(file)

    # Process each row
    for row in csv_reader:
        if row:  # Check if row is not empty
            layer_num.append(int(row[0]))  # First element
            mi_scores.append(float(row[-1]))  # Last element

fig, ax = plt.subplots()
ax.plot(layer_num, mi_scores)
ax.set_title('Mutual Information between word level representations of \n wav2vev and word labels \n with 500 discrete labels')
ax.set_xlabel('wav2vec layer')
ax.set_ylabel('MI Score')
ax.set_xticks(np.arange(12))
plt.show()


mi_results_path = Path(f"{os.pardir}/logs/librispeech_wav2vec_small/mi_word_dev-clean_train-clean-100_5000k.lst")
layer_num = []
mi_scores = []

# Read the .lst file as a CSV
with open(mi_results_path, 'r') as file:
    csv_reader = csv.reader(file)

    # Process each row
    for row in csv_reader:
        if row:  # Check if row is not empty
            layer_num.append(int(row[0]))  # First element
            mi_scores.append(float(row[-1]))  # Last element

fig, ax = plt.subplots()
ax.plot(layer_num, mi_scores)
ax.set_title('Mutual Information between word level representations of \n wav2vev and word labels \n with 5000 labels')
ax.set_xlabel('wav2vec layer')
ax.set_ylabel('MI Score')
ax.set_xticks(np.arange(12))
plt.show()