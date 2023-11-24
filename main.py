from PIL import Image
import cv2
import numpy as np


# ---- ЗАДАНИЕ_1 ----
def prog1():  # Первая программа
    colors = Image.open(input("Введите путь к изображению\n")).getcolors(1)[0][1]  # Получение массива цветов
    colors = colors[:len(colors) - 1] # Удаление лишнего элемента из множества цветов
    n = np.argmax(colors)
    if n == 0: # Вывод взвешенного канала
        print("R")
    elif n == 1:
        print("G")
    elif n == 2:
        print("B")

# Функция обводки и подписи по маске - применяется в 2 и 3 заданиях
def rect(mask, colorname, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Определение контуров по маске
    for c in contours:  # Цикл обработки каждого контура
        x, y, w, h = cv2.boundingRect(c)  # Получение координат для обводки
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Отрисовка обводки
        m = cv2.moments(c)  # Получение точек контура для нахождения координат
        try:  # Отрисовка текста (try except нужны в том случае если контур неправильный и не получается найти координаты центра)
            cv2.putText(frame, f"{colorname} X={round(m['m10'] / m['m00'])} Y={round(m['m01'] / m['m00'])}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except:
            pass


# ---- ЗАДАНИЕ_2 ----
def prog2():  # Вторая программа
    # Открытие исходного файла
    frame = cv2.imread("tello-image.JPG")

    # Установка параметров
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # перевод изображения в цветовое пространство HSV
    lower = np.array([100, 80, 5])  # нижняя граница синего цвета
    upper = np.array([140, 255, 255])  # верхняя граница синего цвета
    colorname = 'BLUE'
    mask = cv2.inRange(hsv, lower, upper)  # Создание маски по параметрам
    rect(mask, colorname, frame)  # Отрисовка по маске

    # То же самое, но используются две маски чтобы захватить красный диапазон hsv с двух сторон цветового круга
    lower = np.array([170, 100, 50])
    upper = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    lower = np.array([0, 100, 50])
    upper = np.array([10, 255, 255])
    mask2 = cv2.inRange(hsv, lower, upper)
    mask = cv2.add(mask1, mask2)
    colorname = 'RED'
    rect(mask, colorname, frame)

    # Запись конечного файла
    cv2.imwrite("tello-image-output.JPG", frame)
    print("Файл записан (tello-image-output.JPG)")


# ---- ЗАДАНИЕ_3 ----
def prog3():
    # Открытие исходного файла
    cap = cv2.VideoCapture("tello-video.MP4")
    # Создание и открытие конечного файла
    out = cv2.VideoWriter("tello-video-output.MP4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:  # Закрытие файлов и завершение программы при достижении конца исходного файла
            cap.release()
            out.release()
            print("Файл записан (tello-video-output.MP4)")
            break

        # Аналогично 2 программе
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # перевод изображения в цветовое пространство HSV
        lower = np.array([100, 80, 5])  # нижняя граница синего цвета
        upper = np.array([140, 255, 255])  # верхняя граница синего цвета
        colorname = 'BLUE'
        mask = cv2.inRange(hsv, lower, upper)  # Создание маски по параметрам
        rect(mask, colorname, frame)  # Функция обводки и подписи по маске - создана в начале документа

        # То же самое, но используются две маски чтобы захватить красный диапазон hsv с двух сторон цветового круга
        lower = np.array([170, 100, 50])
        upper = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower, upper)
        lower = np.array([0, 100, 50])
        upper = np.array([10, 255, 255])
        mask2 = cv2.inRange(hsv, lower, upper)
        mask = cv2.add(mask1, mask2)
        colorname = 'RED'
        rect(mask, colorname, frame)
        # Запись кадра в исходный файл
        out.write(frame)


# ---- ЗАДАНИЕ_4 ----
# Функция загрузки, нормализации картинки и дальшейшего получения числа на ней
def cnn_digits_predict(model, image_file, tf):
    # Открытие файла для распознавания цифры
    img = tf.keras.preprocessing.image.load_img(image_file,
                                                target_size=(28, 28), color_mode='grayscale')
    # Нормализация изображения
    img = np.expand_dims(img, axis=0)  # Перевод изображения в массив numpy
    img = 1 - img / 255.0  # Перевод типа изображения в float32 для облегчения работы модели
    img = img.reshape((1, 28, 28, 1))  # Установка правильной формы массива

    # Получение числа из изображения с помощью модели
    return np.argmax(model.predict([img], verbose=0))
def train4():
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    # Загрузка и нормализация датасета
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Функция сохранения весов в файл
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='cp/cp.ckpt',
                                                     save_weights_only=True,
                                                     verbose=1)

    # Задание формы модели (стек слоев)
    model = tf.keras.models.Sequential([
        # первый слой сети - преобразует формат изображений в одномерный массив 28 * 28 = 784 пикселей
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # второй слой = 128 узлов (нейронов)
        tf.keras.layers.Dense(128, activation='relu'),
        # третий слой (выходной) = 10 узлов (нейронов)
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        # Оптимизатор обучения модели
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        x_train, # Датасет с изображениями
        y_train, # Датасет с расшифровкой изображений
        epochs=50,
        validation_data=(x_test, y_test), # Датасет с тестовыми изображениями
        callbacks=[cp_callback] # Сохранение весов в файл
    )


def prog4():
    import tensorflow as tf
    # Отключение предупреждений в tenserflow мешающих выводу в консоль
    tf.get_logger().setLevel('ERROR')
    # Задание формы модели (стек слоев)
    model = tf.keras.models.Sequential([
        # первый слой сети - преобразует формат изображений в одномерный массив 28 * 28 = 784 пикселей
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # второй слой = 128 узлов (нейронов)
        tf.keras.layers.Dense(128, activation='relu'),
        # третий слой (выходной) = 10 узлов (нейронов)
        tf.keras.layers.Dense(10)
    ])
    # Создание модели
    model.compile()
    # Загрузка весов из чекпоинта модели
    model.load_weights("cp/cp.ckpt")

    # Вывод результатов распознавания графических файлов
    for i in range(1, 6):  # Перебор файлов изображений
        print(f'mnist-test-{i}.png: ' + str(cnn_digits_predict(model, f'mnist-test-{i}.png', tf)))


# ---- ИНТЕРФЕЙС_ПРОВЕРКИ_ЗАДАНИЙ ----
while True:
    uinput = input("Введите номер задания (1-4), 0 для выхода или 5 для переобучения модели из 4 задания\n")
    if uinput == "1":
        prog1()
    elif uinput == "2":
        prog2()
    elif uinput == "3":
        prog3()
    elif uinput == "4":
        prog4()
    elif uinput == "5":
        train4()
    elif uinput == "0":
        print("Выход из программы")
        break
