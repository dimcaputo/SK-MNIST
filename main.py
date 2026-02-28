import os
os.environ["KERAS_BACKEND"] = "torch"
import pandas as pd
import torch
from keras.layers import Input, Rescaling, Dropout
from keras.layers import RandomFlip, RandomTranslation, RandomRotation, RandomZoom
from keras.models import Sequential
from keras.metrics import F1Score
from keras.losses import CategoricalCrossentropy, BinaryFocalCrossentropy
from keras.utils import to_categorical
from keras.optimizers.schedules import CosineDecay
from keras.optimizers import AdamW
from keras.applications import EfficientNetV2B0, VGG16, Xception, MobileNetV2, DenseNet121, ConvNeXtTiny, NASNetMobile
from sklearn.model_selection import train_test_split

from helper_functions import *
import argparse

class_mapping = {
        "akiec": "canc",
        "bcc": "canc",
        "bkl": "nocanc",
        "df": "nocanc",
        "mel": "canc",
        "nv": "nocanc",
        "vasc": "nocanc"
    }

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--size", default=224, type=int)
    parser.add_argument("--label_type", choices=["diseases", "cancer"])
    parser.add_argument("--make_dataset", action="store_true")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--patience", default=50, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--output_name", default="trained_model.pth", type=str)
    parser.add_argument("--model", choices=["EfficientNetV2B0", "VGG16", "Xception", "MobileNetV2", "DenseNet121", "ConvNeXtTiny", "NASNetMobile"])
    args = parser.parse_args()

    df = pd.read_csv('HAM10000_metadata.csv')
    match args.label_type:
        case "diseases":
            dict_label = {k:v for k,v in zip(list(class_mapping.keys()), range(7))}
        case "cancer":
            dict_label = {"nocanc":0, "canc":1}

    if args.make_dataset == True:
        print("Making the dataset...")
        X, y = array_from_images('dataset/', df, dict_label, h=args.size, w=args.size)
        np.savez_compressed(f'X-{args.size}x{args.size}_{args.label_type}', X, allow_pickle=True)
        np.savez_compressed(f'y-{args.size}x{args.size}_{args.label_type}', y, allow_pickle=True)

    print("Loading the dataset...")
    X = np.load(f'X-{args.size}x{args.size}.npz')['arr_0']
    y = np.load(f'y-{args.size}x{args.size}.npz')['arr_0']

    if args.label_type == "diseases":
        class_weights = {i:(len(y)/(7 * sum(y==i))).item() for i in range(7)}

    X_train, X, y_train, y = train_test_split(X, y, stratify=y, test_size=0.3, random_state=38)
    X_test, X_val, y_test, y_val = train_test_split(X, y, stratify=y, test_size=0.5, random_state=38)

    del X,y
    
    match args.label_type:
        case "diseases":
            y_train = to_categorical(y_train, num_classes=7)
            y_val = to_categorical(y_val, num_classes=7)
            y_test = to_categorical(y_test, num_classes=7)

            model = eval(args.model)(
                include_top=True,
                input_shape=X_train.shape[1:],
                weights=None,
                classes=7,
                classifier_activation="softmax",
            )
        case "cancer":
            model = eval(args.model)(
                include_top=True,
                input_shape=X_train.shape[1:],
                weights=None,
                classes=1,
                classifier_activation="sigmoid",
            )
    

    preproc = Sequential([
        RandomFlip(),
        RandomRotation(factor=0.5),
        RandomTranslation(height_factor=0.5, width_factor=0.5),
        RandomZoom(height_factor=0.5, width_factor=0.5)
    ])

    model = Sequential([
        Input(shape=(X_train.shape[1:])),
        Rescaling(scale=1./255),
        preproc,
        model, 
    ])

    lr_schedule = CosineDecay(initial_learning_rate=1e-4, decay_steps=500, alpha=0.1) 
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    match args.label_type:
        case "diseases":
            history020 = compile_and_train(
                model,
                train_data=(X_train, y_train),
                val_data = (X_val, y_val),
                loss=CategoricalCrossentropy(),
                opt=optimizer,
                metrics=[F1Score(average='macro')],
                epochs=args.epochs,
                patience=args.patience, 
                class_weight=class_weights
            )
            try:
                pred, res = get_analysis_cat(model, X_test, y_test)
            except:
                pass
        case "cancer":
            history020 = compile_and_train(
                model,
                train_data=(X_train, y_train),
                val_data = (X_val, y_val),
                loss=BinaryFocalCrossentropy(),
                opt=optimizer,
                metrics=[F1Score(average='macro')],
                epochs=args.epochs,
                patience=args.patience, 
                class_weight=None
            )
            try:
                pred, res = get_analysis_cat(model, X_test, y_test)
            except:
                pass


    torch.save(model, f"{args.output_name}")