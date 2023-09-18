import cv2

for key in {"00001", "00002"}:
    capture = cv2.VideoCapture(f'./data/video/{key}.MOV')

    frameNr = 0

    while (True):
        success, frame = capture.read()

        if success:
            cv2.imwrite(f'./data/frame/{key}/{frameNr}.jpg', frame)
        else:
            break

        frameNr += 1


