import math

import cv2 as cv


def main():
    #phone_video()
    # get_video()
    get_pic()
    # gauss_blur() # чтобы показать картинки, раскомментить внутри метода show_pics
    #get_contur()


def get_contur():
    f_img = open('my_img.txt', 'w')
    f_gx = open('gx.txt', 'w')
    f_gy = open('gy.txt', 'w')
    f_mod = open('mod.txt', 'w')
    f_tan = open('tan.txt', 'w')
    f_angle = open('angle.txt', 'w')
    f_bound = open('bound.txt', 'w')
    img = cv.imread(r'C:\Users\kminin\Pictures\2.jpg', flags=cv.IMREAD_GRAYSCALE)
    blured_img = cv.blur(img, (3, 3))
    w = blured_img.shape[1]
    h = blured_img.shape[0]
    w_1 = 1
    h_1 = 1
    w_2 = w - 1
    h_2 = h - 1
    print(h, " ", w, " ", h_1, " ", h_2)
    my_gx = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
    my_gy = [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]

    img_gx = []     # матрица для x Собеля
    img_gy = []     # матрица для y Собеля
    img_mod_g = []      # матрица модулей градиента
    img_angle_g = []    # матрица направлений градиента
    img_bound = img.copy()      # картинка границ
    for i in range(0, h):
        img_gx.append([0] * w)
        img_gy.append([0] * w)
        img_mod_g.append([0] * w)
        img_angle_g.append([0] * w)
    avg_m = 0

    max_mod_g = -1
    # проход по пикселям картинки
    for i in range(h_1, h_2):
        for j in range(w_1, w_2):
            my_sum1 = 0
            my_sum2 = 0
            g_m_i = 0
            g_m_j = 0
            # свертку для каждого пикселя мутим
            for k in range(i - 1, i + 2):
                for m in range(j - 1, j + 2):
                    my_sum1 += blured_img[k][m] * my_gx[g_m_i][g_m_j]
                    my_sum2 += blured_img[k][m] * my_gy[g_m_i][g_m_j]
                    g_m_j = g_m_j + 1
                g_m_i += 1
                g_m_j = 0
            f_img.write(str(blured_img[i][j]) + " ")
            img_gx[i][j] = my_sum1
            f_gx.write(str(img_gx[i][j]) + " ")
            img_gy[i][j] = my_sum2
            f_gy.write(str(img_gy[i][j]) + " ")
            # считаем максимум модуля по всей картинке, длину градиента и угол для каждого пикселя
            img_mod_g[i][j] = math.sqrt(img_gx[i][j] * img_gx[i][j] + img_gy[i][j] * img_gy[i][j])
            f_mod.write(str(img_mod_g[i][j]) + " ")
            avg_m += img_mod_g[i][j]
            if img_mod_g[i][j] > max_mod_g:
                max_mod_g = img_mod_g[i][j]
            if img_gx[i][j] == 0:
                temp_tg = 0.00001
            else:
                temp_tg = math.tan(img_gy[i][j] / img_gx[i][j])
            f_tan.write(str(temp_tg) + " ")
            if (img_gx[i][j] > 0 and img_gy[i][j] < 0 and temp_tg < -2.414) or \
                    (img_gx[i][j] < 0 and img_gy[i][j] < 0 and temp_tg > 2.414) or \
                    (img_gx[i][j] > 0 and img_gy[i][j] > 0 and temp_tg > 2.414) or \
                    (img_gx[i][j] < 0 and img_gy[i][j] > 0 and temp_tg < -2.414) or \
                    (img_gx[i][j] == 0 and img_gy[i][j] != 0):
                img_angle_g[i][j] = 0
            elif (img_gx[i][j] > 0 and img_gy[i][j] < 0 and temp_tg < -0.414) or \
                    (img_gx[i][j] < 0 and img_gy[i][j] > 0 and temp_tg < -0.414):
                img_angle_g[i][j] = 1
            elif (img_gx[i][j] > 0 and img_gy[i][j] < 0 and temp_tg > -0.414) or \
                    (img_gx[i][j] > 0 and img_gy[i][j] > 0 and temp_tg < 0.414) or \
                    (img_gx[i][j] < 0 and img_gy[i][j] > 0 and temp_tg > -0.414) or \
                    (img_gx[i][j] < 0 and img_gy[i][j] < 0 and temp_tg < 0.414) or \
                    (temp_tg == 0):
                img_angle_g[i][j] = 2
            elif (img_gx[i][j] > 0 and img_gy[i][j] > 0 and temp_tg < 2.414) or \
                    (img_gx[i][j] < 0 and img_gy[i][j] < 0 and temp_tg < 2.414):
                img_angle_g[i][j] = 3
            else:
                img_angle_g[i][j] = -1
            f_angle.write(str(img_angle_g[i][j]) + " ")
        f_img.write("\n")
        f_gx.write("\n")
        f_gy.write("\n")
        f_mod.write("\n")
        f_tan.write("\n")
        f_angle.write("\n")

    # уровни для двойной пороговой фильтрации
    # для груш: размытие 5х5, пороги 5 и 10
    # для геом фигур: размытие 3х3, пороги: 10 и 25
    # для бурундука:размытие 3х3, пороги: 5 и 15
    low_level = max_mod_g * 5 // 100
    high_level = max_mod_g * 15 // 100

    print("low: ", low_level, "; high: ", high_level, ";\navg: ", avg_m / (h * w), "\nmax: ", max_mod_g)

    # проход по пикселям картинки
    for i in range(h_1, h_2):
        for j in range(w_1, w_2):
            img_bound[i][j] = 255
            # проверяем является ли пиксель границей через подавление немаксимумов
            if img_angle_g[i][j] == 0:  # верх и низ
                if img_mod_g[i][j] > img_mod_g[i - 1][j] and img_mod_g[i][j] > img_mod_g[i + 1][j]:
                    img_bound[i][j] = 0
                else:
                    img_bound[i][j] = 255
            elif img_angle_g[i][j] == 1:  # северо-восток и юго-запад
                if img_mod_g[i][j] > img_mod_g[i - 1][j + 1] and img_mod_g[i][j] > img_mod_g[i + 1][j - 1]:
                    img_bound[i][j] = 0
                else:
                    img_bound[i][j] = 255
            elif img_angle_g[i][j] == 2:  # лево и право
                if img_mod_g[i][j] > img_mod_g[i][j + 1] and img_mod_g[i][j] > img_mod_g[i][j - 1]:
                    img_bound[i][j] = 0
                else:
                    img_bound[i][j] = 255
            elif img_angle_g[i][j] == 3:  # северо-запад и юго-восток
                if img_mod_g[i][j] > img_mod_g[i - 1][j - 1] and img_mod_g[i][j] > img_mod_g[i + 1][j + 1]:
                    img_bound[i][j] = 0
                else:
                    img_bound[i][j] = 255

            # двойная пороговая фильтрация
            if img_bound[i][j] == 0:
                if img_mod_g[i][j] < low_level:
                    img_bound[i][j] = 255
                else:
                    if img_mod_g[i][j] <= high_level:
                        # если вокруг нет ни одного соседа, то это не граница
                        if img_mod_g[i - 1][j - 1] != 0 and img_mod_g[i - 1][j] != 0 and img_mod_g[i - 1][j + 1] != 0 \
                                and img_mod_g[i][j - 1] != 0 and img_mod_g[i][j + 1] != 0 \
                                and img_mod_g[i + 1][j - 1] != 0 and img_mod_g[i + 1][j] != 0 \
                                and img_mod_g[i + 1][j + 1] != 0:
                            img_bound[i][j] = 255
            f_bound.write(str(img_bound[i][j]) + " ")
        f_bound.write("\n")

    # выводим картинку и картинку с границами
    f_gx.close()
    f_gy.close()
    f_img.close()
    f_mod.close()
    f_tan.close()
    f_angle.close()
    f_bound.close()
    show_pics(img, "Image", img_bound, "Bounds of object")


def gauss_blur():
    n = 9
    sigma = 2
    g_matr = make_gauss_matr(n, sigma)
    img1, wi, he = get_pic_gauss()
    img2 = img1.copy()
    n_m_2 = n // 2

    w_1 = n_m_2
    h_1 = n_m_2
    w_2 = wi - n_m_2
    h_2 = he - n_m_2
    # 3840 x 2160
    print(w_1, h_1, w_2, h_2)
    # проход по пикселям картинки
    for i in range(w_1, w_2):
        for j in range(h_1, h_2):
            # проход по пикселям внутри квадрата Гаусса
            my_sum = 0
            g_m_i = 0
            g_m_j = 0
            for k in range(i - n_m_2, i + n_m_2 + 1):
                for m in range(j - n_m_2, j + n_m_2 + 1):
                    my_sum += img1[k][m] * g_matr[g_m_i][g_m_j]
                    g_m_j = g_m_j + 1
                g_m_i += 1
                g_m_j = 0
            img2[i][j] = my_sum
    show_pics(img1, "Non blured", img2, "Blured")


def phone_video():
    cap = cv.VideoCapture(2)  # cv.CAP_DSHOW параметр позволяет обратиться к встроенной вебке, но
    cap.set(3, 800)  # все равно не отображается изображение. с телефона норм и без него, только индекс 2
    cap.set(4, 600)
    print(cap.isOpened())
    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv.destroyAllWindows()


def get_pic():
    img1 = cv.imread(r'C:\Users\kminin\Pictures\backgrounds\1.jpg')  # cv.IMREAD_REDUCED_COLOR_4
    cv.namedWindow("Display window", cv.WINDOW_NORMAL)  # cv.WINDOW_NORMAL cv.WINDOW_AUTOSIZE cv.WINDOW_FULLSCREEN
    cv.imshow("Display window", img1)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_pics(my_pic1, win_name1, my_pic2, win_name2):
    cv.namedWindow(win_name1, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name1, my_pic1)
    cv.namedWindow(win_name2, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name2, my_pic2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_video():
    cap = cv.VideoCapture(r'C:\Users\kminin\Pictures\Camera Roll\vid1.mp4', cv.CAP_ANY)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    output = cv.VideoWriter('output_video_from_file.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, frame_size)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow('frame', frame)
        output.write(frame)
        if cv.waitKey(1) & 0xFF == 27:
            break


def get_pic_gauss():
    img = cv.imread(r'C:\Users\kminin\Pictures\2.jpg', flags=cv.IMREAD_GRAYSCALE)
    w = img.shape[0]
    h = img.shape[1]
    return img, w, h


def make_gauss_matr(n, sigma):
    matr = []
    for i in range(0, n):
        matr.append([0] * n)
    ab = n // 2
    print("Matrix size: " + str(len(matr)))
    # print(len(matr[0]))
    for i in range(0, n):
        for j in range(0, n):
            matr[i][j] = 1 / (2 * math.pi * sigma * sigma) * math.pow(math.e,
                                                                      -(math.pow(i - ab, 2) + math.pow(j - ab, 2)) / (
                                                                              2 * sigma * sigma))
    s = 0
    for i in range(0, n):
        s += sum(matr[i])
    print("Sum of elements: " + str(s))
    return matr


if __name__ == "__main__":
    main()
