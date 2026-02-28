import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, saves plots to files
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
import subprocess

from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)
BATCH_SIZE = 64
SEED       = 42
DATA_DIR   = './data'
OUT_DIR    = './outputs'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')

# ─── 1. Download dataset if missing ───────────────────────────────────────────
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_fd2c5151f762387148d28c5a8dd2d08f'
if not os.path.exists(os.path.join(DATA_DIR, 'train_data')):
    print('Preuzimanje dataseta...')
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'alessandrasala79/ai-vs-human-generated-dataset',
        '-p', DATA_DIR, '--unzip'
    ], check=True)
    print('Dataset preuzet i raspakovan.')
else:
    print('Dataset vec postoji.')

# ─── 2. Load data from CSV ────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col=0)
df['full_path'] = df['file_name'].apply(lambda x: os.path.join(DATA_DIR, x))
df['label'] = df['label'].astype(int)

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
n = len(df)
train_df = df.iloc[:int(0.7 * n)]
val_df   = df.iloc[int(0.7 * n):int(0.8 * n)]
test_df  = df.iloc[int(0.8 * n):]

classes     = ['AI', 'Human']
num_classes = 2

print(f'Trening skup:     {len(train_df)} slika')
print(f'Validacioni skup: {len(val_df)} slika')
print(f'Test skup:        {len(test_df)} slika')

def make_dataset(dataframe, shuffle=False):
    paths  = dataframe['full_path'].values
    labels = dataframe['label'].values

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=2000, seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

Xtrain = make_dataset(train_df, shuffle=True)
Xval   = make_dataset(val_df)
Xtest  = make_dataset(test_df)

# ─── 3. Sample images ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, num_classes, figsize=(10, 5))
shown_classes = set()
for images, labels in Xtrain:
    for i in range(len(labels)):
        label = labels[i].numpy()
        if label not in shown_classes:
            axes[label].imshow(images[i].numpy().astype('uint8'))
            axes[label].set_title(classes[label], fontsize=14)
            axes[label].axis('off')
            shown_classes.add(label)
        if len(shown_classes) == num_classes:
            break
    if len(shown_classes) == num_classes:
        break
plt.suptitle('Jedan primerak iz svake klase', fontsize=16)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_sample_images.png', dpi=150)
plt.close()
print('Saved: 01_sample_images.png')

# ─── 4. Class distribution ────────────────────────────────────────────────────
class_counts = {c: 0 for c in classes}
for _, labels in Xtrain.unbatch():
    class_counts[classes[labels.numpy()]] += 1

print('Broj odbiraka po klasama (trening skup):')
for c, count in class_counts.items():
    print(f'  {c}: {count}')

counts = list(class_counts.values())
ratio  = max(counts) / min(counts) if min(counts) > 0 else float('inf')
print(f'Odnos najvece/najmanje klase: {ratio:.2f}')
print('Podaci su priblizno balansirani.' if ratio < 1.5 else 'Podaci NISU balansirani.')

plt.figure(figsize=(8, 5))
plt.bar(class_counts.keys(), class_counts.values(), color=['#4C72B0', '#DD8452'])
plt.xlabel('Klasa')
plt.ylabel('Broj odbiraka')
plt.title('Raspodela odbiraka po klasama (trening skup)')
for i, (c, v) in enumerate(class_counts.items()):
    plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_class_distribution.png', dpi=150)
plt.close()
print('Saved: 02_class_distribution.png')

# ─── 5. Class weights ─────────────────────────────────────────────────────────
all_labels = []
for _, labels in Xtrain.unbatch():
    all_labels.append(labels.numpy())
all_labels = np.array(all_labels)

weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weight = dict(enumerate(weights))
class_weight[0] *= 2.0  # penalize AI misses more
print(f'Class weights (korigovani): {class_weight}')

# ─── 6. Augmentation preview ──────────────────────────────────────────────────
data_augmentation = Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
])

plt.figure(figsize=(12, 6))
for images, _ in Xtrain.take(1):
    img = images[0]
    for i in range(8):
        aug_img = data_augmentation(tf.expand_dims(img, 0))
        plt.subplot(2, 4, i + 1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
plt.suptitle('Primeri augmentacije jedne slike', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_augmentation.png', dpi=150)
plt.close()
print('Saved: 03_augmentation.png')

# ─── 7. Define model ──────────────────────────────────────────────────────────
def cnn_model(num_classes):
    model = Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

model = cnn_model(num_classes)

# ─── 8. Train ─────────────────────────────────────────────────────────────────
es = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    Xtrain,
    epochs=30,
    validation_data=Xval,
    callbacks=[es],
    class_weight=class_weight,
    verbose=1
)

# ─── 9. Save model ────────────────────────────────────────────────────────────
model.save('cnn_model.keras')
print('Model sacuvan: cnn_model.keras')
model.summary()

# ─── 10. Training curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Trening')
ax1.plot(history.history['val_accuracy'], label='Validacija')
ax1.set_title('Tacnost po epohama')
ax1.set_xlabel('Epoha')
ax1.set_ylabel('Tacnost')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Trening')
ax2.plot(history.history['val_loss'], label='Validacija')
ax2.set_title('Gubitak po epohama')
ax2.set_xlabel('Epoha')
ax2.set_ylabel('Gubitak')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_training_curves.png', dpi=150)
plt.close()
print('Saved: 04_training_curves.png')

# ─── 11. Evaluate on test set ─────────────────────────────────────────────────
y_true_test, y_pred_test = np.array([]), np.array([])
for img, lab in Xtest:
    y_true_test = np.concatenate([y_true_test, lab.numpy()])
    y_pred_test = np.concatenate([y_pred_test, np.argmax(model.predict(img, verbose=0), axis=1)])

print(f'\nTacnost modela na test skupu: {100 * accuracy_score(y_true_test, y_pred_test):.2f}%')
print('\nDetaljan izvestaj:')
print(classification_report(y_true_test, y_pred_test, target_names=classes))

y_true_train, y_pred_train = np.array([]), np.array([])
for img, lab in Xtrain:
    y_true_train = np.concatenate([y_true_train, lab.numpy()])
    y_pred_train = np.concatenate([y_pred_train, np.argmax(model.predict(img, verbose=0), axis=1)])

print(f'Tacnost modela na trening skupu: {100 * accuracy_score(y_true_train, y_pred_train):.2f}%')

# ─── 12. Confusion matrix ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

cm_train = confusion_matrix(y_true_train, y_pred_train, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=classes).plot(ax=ax1)
ax1.set_title('Matrica konfuzije - Trening skup')

cm_test = confusion_matrix(y_true_test, y_pred_test, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes).plot(ax=ax2)
ax2.set_title('Matrica konfuzije - Test skup')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_confusion_matrix.png', dpi=150)
plt.close()
print('Saved: 05_confusion_matrix.png')

# ─── 13. Good/bad classified images ──────────────────────────────────────────
all_images, all_true, all_pred = [], [], []
for img_batch, lab_batch in Xtest:
    preds = np.argmax(model.predict(img_batch, verbose=0), axis=1)
    for i in range(len(lab_batch)):
        all_images.append(img_batch[i].numpy().astype('uint8'))
        all_true.append(lab_batch[i].numpy())
        all_pred.append(preds[i])

all_true = np.array(all_true)
all_pred = np.array(all_pred)
correct_idx   = np.where(all_true == all_pred)[0]
incorrect_idx = np.where(all_true != all_pred)[0]
print(f'\nUkupno tacno klasifikovanih:   {len(correct_idx)}')
print(f'Ukupno pogresno klasifikovanih: {len(incorrect_idx)}')

n_show = min(8, len(correct_idx))
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
sample = np.random.choice(correct_idx, n_show, replace=False)
for i, idx in enumerate(sample):
    ax = axes[i // 4][i % 4]
    ax.imshow(all_images[idx])
    ax.set_title(f'Stvarno: {classes[all_true[idx]]}\nPredikcija: {classes[all_pred[idx]]}', fontsize=10)
    ax.axis('off')
plt.suptitle('Primeri DOBRO klasifikovanih slika', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_correct_predictions.png', dpi=150)
plt.close()
print('Saved: 06_correct_predictions.png')

n_show = min(8, len(incorrect_idx))
if n_show > 0:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    sample = np.random.choice(incorrect_idx, n_show, replace=False)
    for i, idx in enumerate(sample):
        ax = axes[i // 4][i % 4]
        ax.imshow(all_images[idx])
        ax.set_title(f'Stvarno: {classes[all_true[idx]]}\nPredikcija: {classes[all_pred[idx]]}', fontsize=10, color='red')
        ax.axis('off')
    for i in range(n_show, 8):
        axes[i // 4][i % 4].axis('off')
    plt.suptitle('Primeri POGRESNO klasifikovanih slika', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/07_wrong_predictions.png', dpi=150)
    plt.close()
    print('Saved: 07_wrong_predictions.png')

print('\nGotovo! Svi rezultati su sacuvani u ./outputs/')
