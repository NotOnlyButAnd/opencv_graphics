import cv2 as cv


def main():
    phone_video()
    # get_video()
    #get_pic()

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
    img1 = cv.imread(r'C:\Users\kminin\Pictures\backgrounds\1.jpg', cv.IMREAD_REDUCED_COLOR_)
    cv.namedWindow("Display window", cv.WINDOW_NORMAL)  # cv.WINDOW_NORMAL cv.WINDOW_AUTOSIZE cv.WINDOW_FULLSCREEN
    cv.imshow("Display window", img1)
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

if __name__ == "__main__":
    main()
