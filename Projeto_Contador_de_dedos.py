import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

mao = mp.solutions.hands
Mao = mao.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = Mao.process(imgRGB)
    Pontosmao = resultado.multi_hand_landmarks
    h, w, _ = img.shape
    ponto = []

    contador_total = 0

    if Pontosmao:
        for mao_num, pontos_mao in enumerate(Pontosmao):
            mpDraw.draw_landmarks(img, pontos_mao, mao.HAND_CONNECTIONS)
            for id, cord in enumerate(pontos_mao.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                ponto.append((cx, cy))

            dedos = [4, 8, 12, 16, 20]

            contador = 0

            for dedo in dedos:
                if ponto[dedo][1] < ponto[dedo - 2][1]:
                    contador += 1

            contador_total += contador

            ponto = []

    cv2.putText(img, str(contador_total), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)

    cv2.imshow("Imagem", img)
    cv2.waitKey(1)
