import pandas as pd
import csv


def read_any(file,
             fmt=None,
             header_rows=5,
             index_col=None,
             **kwargs):
    """
    Универсальный читатель файлов в DataFrame.

    Parameters
    ----------
    path : str or path-like
        Путь к файлу (csv, txt, xls, xlsx и т.д.).
    fmt : str, optional
        Явно заданный формат: 'csv', 'txt', 'excel'.
        Если None — определяется по расширению.
    header_rows : int, default 5
        Число строк для детекции заголовка.
    index_col : str or int, optional
        Имя или номер столбца для индекса.
    **kwargs :
        Дополнительные параметры для pandas.read_*.

    Returns
    -------
    pd.DataFrame
    """
    # 1. Определение формата по расширению если не задан явно
    if fmt is None:
        ext = path.split('.')[-1].lower()
        if ext in ('xls', 'xlsx', 'xlsm', 'ods', 'odt'):
            fmt = 'excel'
        elif ext in ('csv', 'txt'):
            fmt = 'csv'
        else:
            raise ValueError(f"Неизвестный формат: {ext}")

    # 2. Функция детекции разделителя для CSV/TXT
    sep = kwargs.get('sep', None)
    if fmt == 'csv' and sep is None:
        with open(path, 'r', encoding=kwargs.get('encoding', 'utf-8')) as f:
            sample = ''.join([next(f) for _ in range(header_rows)])
            # авто-детект разделителя :contentReference[oaicite:5]{index=5}
            sep = csv.Sniffer().sniff(sample).delimiter

    # 3. Динамическая детекция наличия заголовка
    header = 0  # по умолчанию считаем, что есть
    if fmt in ('csv', 'excel'):
        # прочитаем заголовочные строки в буфер
        sample_df = pd.read_csv(path, nrows=header_rows,
                                sep=sep, header=None, **{'engine': 'python'})
        # если в первой строке все колонки — числа, вероятно, заголовка нет
        if sample_df.iloc[0].apply(lambda x: isinstance(x, (int, float))).all():
            # нет заголовка :contentReference[oaicite:6]{index=6}
            header = None

    # 4. Чтение файла нужным методом
    if fmt == 'excel':
        df = pd.read_excel(path,
                           header=header,
                           index_col=index_col,
                           **kwargs)  # поддерживается reading multiple sheets :contentReference[oaicite:7]{index=7}
    else:  # csv / txt
        # txt ⇒ read_table :contentReference[oaicite:8]{index=8}
        reader = pd.read_table if fmt == 'txt' else pd.read_csv
        df = reader(path,
                    sep=sep,
                    header=header,
                    index_col=index_col,
                    **kwargs)

    return df
