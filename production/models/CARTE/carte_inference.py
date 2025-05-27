import pandas as pd
import numpy as np
from carte_ai import Table2GraphTransformer, CARTERegressor
from pathlib import Path


def get_preprocessor_and_model():
    BASE_DIR = Path(__file__).parent
    model_path_en = str(BASE_DIR/"cc.en.300.bin")
    model_path_ru = str(BASE_DIR/"cc.ru.300.bin")

    preprocessor = Table2GraphTransformer(fasttext_model_path=model_path_en)

    fixed_params = dict()
    fixed_params["num_model"] = 3
    fixed_params["disable_pbar"] = False
    fixed_params["random_state"] = 0
    fixed_params["load_pretrain"] = True
    fixed_params["device"] = "cpu"
    fixed_params["n_jobs"] = 10
    fixed_params["pretrained_model_path"] = "CARTE/carte_weights.pt"

    estimator = CARTERegressor(**fixed_params)

    return preprocessor, estimator


def get_prediction(preprocessor: Table2GraphTransformer, model: CARTERegressor, data: pd.DataFrame, min_value: int = 0,
                   max_value: int = 10, integer_grades_flag=1, clipping=True, eps=10e-6):
    for i in range(data.shape[1]):
        if not data.iloc[:, i].isna().sum():
            continue

        y_train = data.iloc[:, i][data.iloc[:, i].notna()]

        if not clipping:
            y_prev = y_train
            y_train = np.log((y_train-min_value+eps)/(100-max_value+eps))

        X_train = data[data.iloc[:, i].notna()].drop(columns=data.columns[i])
        X_test = data[data.iloc[:, i].isna()].drop(columns=data.columns[i])

        X_graph_train = preprocessor.fit_transform(X_train, y=y_train)
        X_graph_test = preprocessor.transform(X_test)

        model.fit(X_graph_train, y_train)
        if len(X_test) == 1:
            pred = model.predict(X_graph_test+X_graph_test)[:1]
        else:
            pred = model.predict(X_graph_test)

        na_idx = data.index[data.iloc[:, i].isna()]

        if clipping:
            pred = np.floor(
                pred + 0.5) if integer_grades_flag else np.round(pred, 2)
            data.iloc[na_idx, i] = np.clip(pred, min_value, max_value)
        else:
            p = (y_prev - min_value) / (max_value - min_value)
            p_adj = np.minimum(np.maximum(p, eps), 1-eps)
            t = np.log(p_adj / (1 - p))
            pred = min_value + (max_value - min_value) / (1 + np.exp(-t))
            pred = np.floor(
                pred + 0.5) if integer_grades_flag else np.round(pred, 2)
            data.iloc[na_idx, i] = pred
