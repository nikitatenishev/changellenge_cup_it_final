import gc
from typing import Dict, List, Tuple, Union

import implicit
import lightgbm
import numpy as np
import pandas as pd
import scipy
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tqdm import notebook


def make_target(
        filename: str
) -> pd.core.frame.DataFrame:
    """
    Функция предназначена для создания целевой переменной
    :filename: имя файла с данными
    :return: датафрейм, в котором содержится таргет
    """
    data = pd.read_csv(filename)
    data.drop("card_uk", axis=1, inplace=True)

    data.drop_duplicates(inplace=True)
    data = data[data["cardapp_date"] != "1900-01-01"]
    data.reset_index(drop=True, inplace=True)

    target_df = data.groupby("owner_id").agg(
        lst=("holder_id", list)
    ).reset_index()

    res = []
    for owner in notebook.tqdm(target_df.iterrows()):
        tmp = owner[1]["lst"]
        res.append(
            len(
                tuple(filter(lambda x: x != owner[1]["owner_id"], tmp))
            )
        )

    target_df["tmp"] = res
    target_df["target"] = target_df["tmp"].apply(
        lambda x: 1 if x > 0 else 0
    )
    target_df.drop(["lst", "tmp"], axis=1, inplace=True)

    data = data.merge(target_df, on="owner_id", how="left")

    del target_df
    gc.collect()

    return data


def reduce_mem_usage(
        df: pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """
    Функция уменьшает расход памяти
    :df: исходный датафрейм
    :возвращает: датафрейм с преобразованными типами данными
    """
    st_mem = df.memory_usage().sum() / 1024**2
    print(f"[было]: {round(st_mem, 3)} Мб")

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (
                    (c_min > np.iinfo(np.int8).min) and
                    (c_max < np.iinfo(np.int8).max)
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    (c_min > np.iinfo(np.int16).min) and
                    (c_max < np.iinfo(np.int16).max)
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    (c_min > np.iinfo(np.int32).min) and
                    (c_max < np.iinfo(np.int32).max)
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    (c_min > np.iinfo(np.int64).min) and
                    (c_max < np.iinfo(np.int64).max)
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    (c_min > np.finfo(np.float16).min) and
                    (c_max < np.finfo(np.float16).max)
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    (c_min > np.finfo(np.float32).min) and
                    (c_max < np.finfo(np.float32).max)
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"[стало]: {round(end_mem, 3)} Мб")
    print(f"[оптимизировано]: {round(100 * (st_mem - end_mem) / st_mem, 3)} %")
    return df


def get_chi_stat(data: np.ndarray) -> List[float]:
    """
    Рассчитывает статистику хи-квадрат
    :data: входной массив наблюдаемых частот
    :возвращает: список хи-квадрат статистик
    """
    chi_stats = []
    for i in range(data.shape[1]):
        vals = data[:, i]
        f_exp = vals.sum() / 2
        chi_stats.append(
            (((vals - f_exp) ** 2) / f_exp).sum()
        )
    return chi_stats


def provide_chi_squared_test(data: np.ndarray) -> List[float]:
    """
    Функция предназначена для проведения теста хи-квадрат
    таблиц сопряжённости размерностей больших 2 на 2
    :data: исходный массив наблюдаемых частот
    :возвращает: p-value
    """
    df = data.shape[1] - 1
    chi_stats = get_chi_stat(data)
    pvalues = [
        scipy.stats.chi2.pdf(val, df=df)
        for val in chi_stats
    ]
    return pvalues


def check_distribution(data: np.ndarray) -> float:
    """
    Функция выполняет проверку на нормальность
    распределения данных
    :data: исходный массив данных
    :возвращает: число, говорящее нормально, если 0,
    и ненормально, если 1, распределены данные
    Если 0.5, то нужно дополнить анализировать
    данные
    """
    kstest_res = 1 if scipy.stats.kstest(
        data,
        scipy.stats.norm.cdf
    ).pvalue < 0.05 else 0
    shapiro_res = 1 if scipy.stats.shapiro(data).pvalue < 0.05 else 0
    return (kstest_res + shapiro_res) / 2


def check_homogenity(
    arr1: np.ndarray,
    arr2: np.ndarray
) -> int:
    """
    Функия реализует проверку гомогенности
    дисперсий двух массивов
    :arr1: первый исходный массив
    :arr2: второй исходный массив
    :возвращает: 1, если дисперсии неоднороды,
    0 - в противном случае
    """
    levene_res = scipy.stats.levene(arr1, arr2).pvalue
    return 1 if levene_res < 0.05 else 0


def mannwhitneyu_test(
    arr1: np.ndarray,
    arr2: np.ndarray
) -> int:
    """
    Функия реализует тест Манна-Уитни
    для двух массивов
    :arr1: первый исходный массив
    :arr2: второй исходный массив
    :возвращает: 1, если различия есть,
    0 - в противном случае
    """
    res = scipy.stats.mannwhitneyu(arr1, arr2).pvalue
    return 1 if res < 0.05 else 0


def load_profile(path: str) -> List[str]:
    """
    Функция реализует загрузку профиля пользователя
    :path: путь до файла
    :возвращает: список признаков
    """
    with open(path, 'r') as file:
        cols = list(map(
            lambda x: x.replace('\n', ''),
            file.readlines()
        ))
    print(f"профиль содержит [{len(cols)}] признака(-ов)")
    return cols


def plot_values(
        mode: int,
        data: Dict[str, List[float]]
) -> None:
    """
        Функция предзаначена для визуализации результатов
        обучения модели
        :mode: режим: 0 - минимальные значения,
        1 - максимальные значения, 2 - средние значения
        :data: словарь с данными обучения
    """
    if (mode > 2) or (mode < 0):
        raise ValueError("Неверные значения для mode")
    title = "Минимальные" if (mode == 0) else (
        "Максимальные" if (mode == 1) else "Средние"
    )
    plt.plot(data["lr"][mode::3], label="lr")
    plt.plot(data["dt"][mode::3], label="dt")
    plt.plot(data["rf"][mode::3], label="rf")
    plt.plot(data["cb"][mode::3], label="cb")
    plt.plot(data["xgb"][mode::3], label="xgb")
    plt.plot(data["lgbm"][mode::3], label="lgbm")
    plt.legend(loc="best")
    plt.title(f"{title} значения ф-меры")
    plt.xlabel("№ датасета")
    plt.ylabel("ф-мера")
    plt.show()


def compare_plot(
        first_name_of_model: str,
        second_name_of_model: str,
        data: Dict[str, List[float]]
) -> None:
    """
    Функция выводит графики, сравнивающие перформанс
    двух моделей
    :first_name_of_model: название первой модели
    :second_name_of_model: название второй модели
    :data: словарь с данными обучения
    """
    plt.figure(figsize=(14, 10))
    plt.suptitle("Сравнение catboost и lightgbm")

    plt.subplot(2, 2, 1)
    plt.plot(data[first_name_of_model][::3], label=first_name_of_model)
    plt.plot(data[second_name_of_model][::3], label=second_name_of_model)
    plt.legend(loc="best")
    plt.title("Минимальные значения ф-меры")
    plt.xlabel("№ датасета")
    plt.ylabel("ф-мера")

    plt.subplot(2, 2, 2)
    plt.plot(data[first_name_of_model][1::3], label=first_name_of_model)
    plt.plot(data[second_name_of_model][1::3], label=second_name_of_model)
    plt.legend(loc="best")
    plt.title("Максимальные значения ф-меры")
    plt.xlabel("№ датасета")
    plt.ylabel("ф-мера")

    plt.subplot(2, 2, 3)
    plt.plot(data[first_name_of_model][2::3], label=first_name_of_model)
    plt.plot(data[second_name_of_model][2::3], label=second_name_of_model)
    plt.legend(loc="best")
    plt.title("Средние значения ф-меры")
    plt.xlabel("№ датасета")
    plt.ylabel("ф-мера")

    plt.show()


def upsample(
        features: pd.core.frame.DataFrame,
        target: pd.core.series.Series,
        repeat: int
) -> Tuple[pd.core.frame.DataFrame, pd.core.series.Series]:
    """
    Функция предназначена для сэмплирования минорного класса
    путём дублирования строк в датасете
    :features: датафрейм с признаками
    :target: столбец с таргетом
    :repeat: кол-во повторений
    :возвращает: кортеж из pandas датафрейма и серии
    """
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)

    return features_upsampled, target_upsampled


def validate(
    x: pd.core.frame.DataFrame,
    y: pd.core.series.Series,
    params: Dict[str, Union[str, bool, int, float]]
) -> List[float]:
    """
    Функция реализует валидацию модели
    :x: датафрейм с признаками
    :y: столбце с таргетом
    :params: словарь с параметрами классификатора
    :возвращает: список ф-меры на валидации
    """
    clf = lightgbm.LGBMClassifier(**params)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)

    smote_scores = []
    for train_idx, val_idx in notebook.tqdm(kfold.split(x, y)):
        x_train, x_val = x.loc[train_idx], x.loc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf.fit(x_train, y_train)
        smote_scores.append(
            round(f1_score(y_val, clf.predict(x_val), average="binary"), 7)
        )
    return smote_scores


def check_profile(
        cols: List[str],
        data: pd.core.frame.DataFrame
) -> List[float]:
    """
    Функция предназначена для тестирования профилей
    пользователя
    :cols: список со столбцами профиля
    :data: датафрейм с признаками
    :возвращает: список значений ф-меры
    """
    df = data[cols + ["target"]]
    smote = SMOTE(
        sampling_strategy="minority",
        k_neighbors=10,
        random_state=11
    )

    df_res, target_res = smote.fit_resample(df[cols], df["target"])

    df_res["target"] = target_res

    scores = validate(
        df_res[cols],
        df_res["target"],
        params={"random_state": 11, "verbose": 0, "force_col_wise": True}
    )
    print(f"[среднее значение]: {round(np.mean(scores), 7)}")
    return scores


def get_embdes(
        data: pd.core.frame.DataFrame,
        target: pd.core.series.Series,
        factors: int
) -> pd.core.frame.DataFrame:
    """
    Функция возвращает датафрейм с эмбеддингами,
    полученными с помощью матричной факторизации
    :data: датафрейм с признаками
    :target: столбец с таргетом
    :factors: кол-во факторов
    :возвращает: датафрейм с эмбеддингами
    """
    sparse = scipy.sparse.csr_matrix(data.values)

    als_model = implicit.als.AlternatingLeastSquares(
        iterations=30,
        regularization=0.01,
        factors=factors,
        use_gpu=implicit.gpu.HAS_CUDA,
        num_threads=12,
        random_state=11
    )
    als_model.fit(sparse, show_progress=True)

    user_embdes = als_model.user_factors

    als_df = pd.DataFrame(user_embdes)
    als_df.columns = [f"als_{i}" for i in range(user_embdes[0].shape[0])]
    als_df["target"] = target
    return als_df


def check_profile_without_smote(
        cols: List[str],
        data: pd.core.frame.DataFrame
) -> List[float]:
    """
    Функция предназначена для тестирования профилей
    пользователя
    :cols: список со столбцами профиля
    :data: датафрейм с признаками
    :возвращает: список значений ф-меры
    """
    df = data[cols + ["target"]]

    scores = validate(
        df[cols],
        df["target"],
        params={"random_state": 11, "verbose": 0, "force_col_wise": True}
    )
    print(f"[среднее значение]: {round(np.mean(scores), 7)}")
    return scores


def downsampling_validate(
        dfs: List[pd.core.frame.DataFrame],
        cols: List[str],
        params: Dict[str, Union[str, bool, int, float]]
) -> Tuple[float]:
    """
    Функция предназначена для комплексной оценки
    downsampling подхода
    :data: список датафреймов
    :cols: список признаков датафрейма
    :params: словарь с параметрами классификатора
    :возвращает: кортеж метрик
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
    clf = lightgbm.LGBMClassifier(**params)

    f_down_scores, roc_auc_down_scores = [], []
    pre_down_scores, rec_down_scores = [], []

    for df in notebook.tqdm(dfs):
        y, x = df["target"], df[cols]
        tmp1, tmp2, tmp3, tmp4 = [], [], [], []

        for train_idx, val_idx in kfold.split(x, y):
            x_train, x_val = x.loc[train_idx], x.loc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf.fit(x_train, y_train)

            tmp1.append(
                round(f1_score(y_val, clf.predict(x_val), average="binary"), 7)
            )
            tmp2.append(
                round(roc_auc_score(y_val, clf.predict(x_val)), 7)
            )
            tmp3.append(
                round(precision_score(y_val, clf.predict(x_val)), 7)
            )
            tmp4.append(
                round(recall_score(y_val, clf.predict(x_val)), 7)
            )

        f_down_scores.append(np.mean(tmp1))
        roc_auc_down_scores.append(np.mean(tmp2))
        pre_down_scores.append(np.mean(tmp3))
        rec_down_scores.append(np.mean(tmp4))

    return (
        round(np.mean(f_down_scores), 7),
        round(np.mean(roc_auc_down_scores), 7),
        round(np.mean(pre_down_scores), 7),
        round(np.mean(rec_down_scores), 7)
    )


def smote_validate(
    x: pd.core.frame.DataFrame,
    y: pd.core.series.Series,
    params: Dict[str, Union[str, bool, int, float]]
) -> Tuple[float]:
    """
    Функция предназначена для комплексной оценки
    upsampling подхода
    :x: исходный датафрейм
    :y: столбец с таргетом
    :params: словарь с параметрами классификатора
    :возвращает: кортеж метрик
    """
    clf = lightgbm.LGBMClassifier(**params)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
    smote = SMOTE(
        sampling_strategy="minority",
        k_neighbors=10,
        random_state=11
    )

    x_res, y_res = smote.fit_resample(x, y)

    f_up_scores, roc_auc_up_scores = [], []
    pre_up_scores, rec_up_scores = [], []
    for train_idx, val_idx in notebook.tqdm(kfold.split(x_res, y_res)):
        x_train, x_val = x_res.loc[train_idx], x_res.loc[val_idx]
        y_train, y_val = y_res[train_idx], y_res[val_idx]

        clf.fit(x_train, y_train)
        f_up_scores.append(
            round(f1_score(y_val, clf.predict(x_val), average="binary"), 7)
        )
        roc_auc_up_scores.append(
            round(roc_auc_score(y_val, clf.predict(x_val)), 7)
        )
        pre_up_scores.append(
            round(precision_score(y_val, clf.predict(x_val)), 7)
        )
        rec_up_scores.append(
            round(recall_score(y_val, clf.predict(x_val)), 7)
        )

    return (
        round(np.mean(f_up_scores), 7),
        round(np.mean(roc_auc_up_scores), 7),
        round(np.mean(pre_up_scores), 7),
        round(np.mean(rec_up_scores), 7)
    )
