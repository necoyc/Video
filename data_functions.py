import matplotlib.pyplot as plt
import os
import numpy as np
import openpyxl as xl

EXCEL_BOOK_PATH = 'test1'

def save_plot(time_values: np.ndarray, y_values: np.ndarray, folder_name: str, filename: str, x_lable: str, y_label: str):
    """
    Сохраняет график в файл.
    
    :param time_values: Массив значений времени.
    :param y_values: Массив значений y.
    :param folder_name: Имя папки для сохранения файла.
    :param filename: Имя файла.
    :param x_lable: Подпись оси X.
    :param y_label: Подпись оси Y.
    """
    #добавить функцию для сглаживание данных
    plt.plot(time_values, y_values)
    plt.xlabel(x_lable)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(folder_name, f"{filename}.png"), dpi=300)
    plt.close()
def save_data_to_text(time_values: np.ndarray, y_values: np.ndarray, folder_name: str, filename: str) -> np.ndarray:
    """
    Сохраняет данные в текстовый файл и возвращает среднюю глубину.
    
    :param time_values: Массив значений времени.
    :param y_values: Многомерный массив значений y.
    :param folder_name: Имя папки для сохранения файла.
    :param filename: Имя файла.
    :return: Средняя глубина.
    """
    os.makedirs(folder_name, exist_ok=True)
    data_path = os.path.join(folder_name, f"{filename}.txt")
    
    average_depth = np.mean(y_values, axis=0)

    with open(data_path, "w") as file:
        for i in range(len(y_values[0])):
            column_values = [y[i] for y in y_values]
            file.write(f"{time_values[i]:.4f} {' '.join(map(str, column_values))}\n")

    wb = xl.Workbook()
    ws = wb.active
    
    for i in range(len(time_values)):
        ws.cell(row=i + 1, column=1, value=time_values[i])
        ws.cell(row=i + 1, column=2, value=str(y_values[i]))

    os.makedirs(folder_name, exist_ok=True)
    wb.save(os.path.join(folder_name, filename + '.xlsx'))
        


    return average_depth
def process_data(time_values: np.ndarray, y_values: np.ndarray, folder_name: str, filename: str, x_lable: str, y_label: str):
    """
    Обрабатывает данные, сохраняет их в текстовый файл и строит график.
    
    :param time_values: Массив значений времени.
    :param y_values: Многомерный массив значений y.
    :param folder_name: Имя папки для сохранения файла.
    :param filename: Имя файла.
    :param x_lable: Подпись оси X.
    :param y_label: Подпись оси Y.
    """
    average_depth = save_data_to_text(time_values, y_values, folder_name, filename)
    save_plot(time_values, average_depth, folder_name, filename, x_lable, y_label)