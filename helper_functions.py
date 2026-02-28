import numpy as np
from PIL import Image
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

def array_from_images(folder, df_metadata, dict_of_labels, h=224, w=224, channels=3):
    # Create an array of images and labels the size of the number of pictures
    nb_files = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            nb_files += 1
    array = np.zeros(shape=(nb_files, h, w, channels))
    labels = np.zeros(shape=(nb_files,))

    # Check the name and fill array and labels
    df_metadata = df_metadata.set_index('image_id', drop=True)
    count = 0

    # In case of binary use
    class_mapping = {
        "akiec": "canc",
        "bcc": "canc",
        "bkl": "nocanc",
        "df": "nocanc",
        "mel": "canc",
        "nv": "nocanc",
        "vasc": "nocanc"
    }

    for root, dirs, files in os.walk(folder):
        for file in files:
            with Image.open(os.path.join(root, file)) as im:
                array[count,:,:,:] = np.asarray(im.resize((h,w)))
                try:
                    labels[count] = dict_of_labels[class_mapping[df_metadata.loc[file.strip('.jpg'), 'dx']]]
                except:
                    raise TypeError('NOPE')
                    # labels[count] = dict_of_labels[df_metadata.loc[file.strip('.jpg'), 'dx']]
                count += 1
                if count%100 == 0:
                    print(f'{count} images were processed')
    return array, labels

def get_earlystopping(metric_name, patience=10):
    early_stopping = EarlyStopping(
    monitor=f'val_{metric_name}',
    mode='max',
    patience=patience,
    verbose=1,
    restore_best_weights=True)
    return early_stopping

def plot_learning_curves(model):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    try:
        ax[0].plot(model.history.history['val_f1_score'], label='val_f1_score')
        ax[0].plot(model.history.history['f1_score'], label='f1_score')
        ax[0].legend()
    except:
        ax[0].plot(model.history.history['val_avg_prec'], label='val_avg_prec')
        ax[0].plot(model.history.history['avg_prec'], label='avg_prec')
        ax[0].legend()
    ax[1].plot(model.history.history['val_loss'], label='val_loss')
    ax[1].plot(model.history.history['loss'], label='loss')
    ax[1].legend()
    fig.show;

def get_analysis_cat(model, testX, testy):
    plot_learning_curves(model)
    loss, f1 = model.evaluate(testX, testy)
    print(f'The model gave')
    print(f'Loss: {loss:.2f}')
    print(f'F1 Macro: {f1:.2f}')
    # print(f'Avg Prec: {avg_prec:.2f}')
    predy = model.predict(testX)
    resy = to_categorical(np.argmax(predy, axis=1), num_classes=7)
    print(classification_report(testy,resy))
    return predy, resy

def get_analysis(model, testX, testy):
    plot_learning_curves(model)
    loss, avg_prec = model.evaluate(testX, testy)
    print(f'The model gave')
    print(f'Loss: {loss:.2f}')
    # print(f'F1 Macro: {f1:.2f}')
    print(f'Avg Prec: {avg_prec:.2f}')
    predy = model.predict(testX)
    resy = (predy > 0.5).astype(int)
    print(classification_report(testy,resy))
    return predy, resy

def compile_and_train(model, train_data, val_data, loss, opt, metrics, epochs, patience=None, steps=None, class_weight=None):
    model.compile(loss=loss,
                optimizer=opt,
                metrics=metrics)

    model.summary()

    if patience != None:
        model.fit(
            train_data[0],
            train_data[1],
            epochs=epochs,
            validation_data=(val_data[0], val_data[1]),
            callbacks=[get_earlystopping(metrics[0].name, patience)],
            steps_per_epoch=steps,
            class_weight=class_weight
            )
    else:
        model.fit(
            train_data[0],
            train_data[1],
            epochs=epochs,
            validation_data=(val_data[0], val_data[1]),
            steps_per_epoch=steps,
            class_weight=class_weight
            )

    return model