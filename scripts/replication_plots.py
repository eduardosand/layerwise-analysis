import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

parent_directory = Path('/home/eduardo/PycharmProjects/layerwise-analysis/logs/')

small_models = ['wav2vec_small', 'hubert_base_ls960']
large_models = ['wav2vec_libri960_big', 'hubert_large_ll60k']
labels = ['wav2vec', 'hubert']
models = [small_models, large_models]
linestyles = ['-','--']
markers = ['o', '^']
def cca_plot(exp_titles):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    for i, ax in enumerate(axes):
        for k, exp_title in enumerate(exp_titles):
            if len(exp_titles) > 1:
                add_label = '-' + exp_title.replace("cca_","")
                if 'prosody' in add_label:
                    add_label = '- prominence'
            else:
                add_label = ''

            models_ins = models[i]
            for j, model in enumerate(models_ins):
                cca_word_results = parent_directory / f'libriTTS_{model}' / f'{exp_title}_dev-clean_sample1.json'
                with open(cca_word_results, 'r') as file:
                    cca_word_results_data = json.load(file)
                layer_x = []
                cca_scores = []
                for key in cca_word_results_data.keys():
                    layer_x.append(float(key)+1)
                    cca_scores.append(cca_word_results_data[key])

                if len(exp_titles) == 1:
                    color_ind = j
                    colors_prosody = ['green', 'purple']
                else:
                    color_ind = k
                    colors_prosody = ['red', 'blue']
                ax.plot(layer_x, cca_scores, label=labels[j] + add_label, marker=markers[j], linestyle=linestyles[j],
                        color=colors_prosody[color_ind])
                ax.set_ylabel('CCA score')
                if i == 0:
                    ax.set_xticks(np.arange(12)+1)
                else:
                    ax.set_xticks(np.arange(1,24,2)+1)
                    ax.set_xlabel('Layer')
                ax.legend(bbox_to_anchor=(1.05, 0.85), loc='upper left')
    plt.suptitle(f'CCA between word level representations and {exp_title.replace("cca_","")} label')
    plt.tight_layout()
    plt.show()

cca_plot(["cca_word"])
cca_plot(["cca_boundary", "cca_prosody"])