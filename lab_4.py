import time

import cv2 as cv2


def main():
    print("Hello from motions_detector!")
    #detector()
    #get_video()

    url = r'http://212.192.148.155:4747/video'
    ip_detector(url)
    #ip_camera(url)


def detector():
    cap = cv2.VideoCapture(r'C:\Users\kminin\Documents\000 Clouds\YandexDisk\обработка мультимедиа\ЛР4_main_video.mov', cv2.CAP_ANY)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    n = 5
    sigma = 5
    square = 50
    new_frame = cv2.GaussianBlur(gray, (n, n), sigma)

    h = len(gray)
    w = len(gray[0])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("result.mov", fourcc, 25, (w, h))

    # cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)

    while (True):
        cv2.imshow('frame', frame)

        old_frame = new_frame
        ret, frame = cap.read()
        if not (ret):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.GaussianBlur(gray, (n, n), sigma)

        frame_dif = cv2.absdiff(old_frame, new_frame)
        retval, frame_dif = cv2.threshold(frame_dif, 50, 255, cv2.THRESH_BINARY)    # 60, 127

        # RETR_EXTERNAL ивлекает только крайние внешние контуры
        # CHAIN_APPROX_SIMPLE -
        # сжимает горизонтальные, вертикальные и диагональные сегменты и оставляет только их конечные точки.
        # Например, прямоугольный контур, расположенный справа вверх, кодируется 4 точками.
        # CHAIN_APPROX_NONE -
        # сохраняет абсолютно все точки контура. То есть любые 2 последующие точки (x1,y1) и (x2,y2) контура
        # будут либо соседями по горизонтали, вертикали или диагонали, то есть max(abs(x1-x2),abs(y2-y1))==1.
        contours, hierarchy = cv2.findContours(frame_dif, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            if square < cv2.contourArea(i):
                # если удовл
                video_writer.write(frame)
                # continue

        if cv2.waitKey(2) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    get_video()


def ip_detector(url):
    ip_video = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    ret, frame = ip_video.read()

    old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h = len(old_frame)
    w = len(old_frame[0])

    n = 5
    sigma = 5
    square = 20
    blur_old_frame = cv2.GaussianBlur(old_frame, (n, n), sigma)


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer_all = cv2.VideoWriter("all_video.mov", fourcc, 30, (w, h))
    video_writer_moves = cv2.VideoWriter("moves.mov", fourcc, 30, (w, h))

    # cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)


    while (True):
        cv2.imshow('frame', frame)

        old_frame = blur_old_frame
        ret, frame = ip_video.read()

        t = time.localtime()
        curr_time_str = time.strftime("%H:%M:%S %d.%m.%Y",t)
        frame = cv2.putText(frame, curr_time_str, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 205, 50), 2)

        video_writer_all.write(frame)
        video_writer_all.write(frame)
        #video_writer_all.write(frame)

        if not (ret):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_old_frame = cv2.GaussianBlur(gray, (n, n), sigma)

        frame_dif = cv2.absdiff(old_frame, blur_old_frame)
        retval, frame_dif = cv2.threshold(frame_dif, 50, 255, cv2.THRESH_BINARY)  # 60, 127

        contours, hierarchy = cv2.findContours(frame_dif, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            if square < cv2.contourArea(i):
                # если удовл
                video_writer_moves.write(frame)
                # continue

        if cv2.waitKey(2) & 0xFF == 27:
            break
    ip_video.release()
    cv2.destroyAllWindows()

   #get_video()


def get_video():
    cap = cv2.VideoCapture(r'C:\Users\kminin\Documents\pyCharm projects\graphics2\result.mov', cv2.CAP_ANY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def ip_camera(url):
    ip_video = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    while True:
        ret, frame = ip_video.read()
        if not ret:
            break
        cv2.imshow('IP Video', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
