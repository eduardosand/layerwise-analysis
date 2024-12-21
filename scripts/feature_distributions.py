import os
from pathlib import Path
import matplotlib.pyplot as plt

feature_sample_dir = Path(f"{os.pardir}/results/libriTTS/wav2vec_small/libriTTS_dev-clean_sample1/contextualized/word_level")
prominence_labels = feature_sample_dir / f"prominence_labels_0.lst"
boundary_labels = feature_sample_dir / f"boundary_labels_0.lst"

with open(prominence_labels, "r") as prominence_labels_file:
    prominence_labels_data = [float(line) for line in prominence_labels_file.read().splitlines()]
    #prominence_labels_file.read().splitlines().astype(float)

fig, ax = plt.subplots()
ax.hist(prominence_labels_data, bins=25)
ax.set_title("Distribution of continuous prominence labels")
# ax.set_xticks([0,1,2])
plt.show()

with open(boundary_labels, "r") as boundary_labels_file:
    boundary_labels_data = [float(line) for line in boundary_labels_file.read().splitlines()]

fig, ax = plt.subplots()
ax.hist(boundary_labels_data, bins=25)
ax.set_title("Distribution of continuous phrase boundary labels")
# ax.set_xticks([0,1,2])
plt.show()