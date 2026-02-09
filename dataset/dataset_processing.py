import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_stratified_split(data_dir, output_dir, split=(0.6, 0.2, 0.2)):
    # 1. Collect all file paths and their labels
    data = []
    for class_name in os.listdir(data_dir):
        class_path = Path(data_dir) / class_name
        if class_path.is_dir():
            for img in class_path.glob('*.jpg'):
                data.append((str(img), class_name))

    paths, labels = zip(*data)

    # 2. Perform Stratified Splits
    # First split: Train vs Temp (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(split[1] + split[2]), stratify=labels, random_state=42
    )

    # Second split: Val vs Test from the Temp set
    # Ratio needs to be adjusted (e.g., 0.2 is half of 0.4)
    val_ratio = split[1] / (split[1] + split[2])
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_ratio), stratify=temp_labels, random_state=42
    )

    # 3. Save files to folders
    dataset_splits = {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }

    for split_name, (split_paths, split_labels) in dataset_splits.items():
        for path, label in zip(split_paths, split_labels):
            # Create destination path: output/train/class_name/image.jpg
            dest_dir = Path(output_dir) / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest_dir / Path(path).name)

    print(f"✅ Data split complete. Files saved to: {output_dir}")

    # 3. Visualization Data
    plot_data = {
        'Train': train_labels,
        'Val': val_labels
    }
    
    return plot_data, (train_paths, val_paths, test_paths)

src_data = "../../PeopleArt-master/JPEGImages"
plot_counts, splits = create_stratified_split(src_data, output_dir="../../PeopleArt_imagenet_splits")
# 4. Generate the Bar Graph
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

classes = sorted(list(set(plot_counts['Train'])))
train_counts = [plot_counts['Train'].count(c) for c in classes]
val_counts = [plot_counts['Val'].count(c) for c in classes]

x = np.arange(len(classes))
width = 0.35

ax.bar(x - width/2, train_counts, width, label='Train', color='#4A90E2')
ax.bar(x + width/2, val_counts, width, label='Val', color='#F5A623')

ax.set_ylabel('Number of Images')
ax.set_title('Stratified Split: Image Counts per Class')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()