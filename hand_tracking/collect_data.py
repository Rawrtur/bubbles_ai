import cv2
import mediapipe as mp
import csv
import os

from datetime import datetime

FILE = "hand_tracking/examples/hand_data.csv"
SAMPLES_PER_PRESS = 1

# Tasten 0, 1 und 2 stehen für Gestenklassen.
GESTURES = {
    ord("0"): 0,  # Faust
    ord("1"): 1,  # Offene Hand
    ord("2"): 2,  # Zeigefinger
    ord("3"): 3,  # OK
    ord("4"): 4,  # cross
    ord('5'): 5,  # peace
    ord("6"): 6,  # thump up
}

class Data_Collector():
    def __init__(self, data:str)->None:
        self.__mp_hands = mp.solutions.hands
        self.__mp_draw = mp.solutions.drawing_utils
        self.__file: str = data
        # CSV-Datei initialisieren, falls sie noch nicht existiert
        if not os.path.exists(data):
            with open(data, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["label"] + 
                    [f"v{i}" for i in range(63)]
                )
        self.__cap = cv2.VideoCapture(0) # erste angeschlossene Kamera verwenden (0->Gerätekamera)

    def start(self)->None:
        with self.__mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            with open(self.__file, "a", newline="") as f:
                writer = csv.writer(f)

                while self.__cap.isOpened():
                    ok, frame = self.__cap.read()
                    if not ok:
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )
                    result = hands.process(rgb)

                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]

                        self.__mp_draw.draw_landmarks(
                            frame, hand,
                            self.__mp_hands.HAND_CONNECTIONS
                        )

                    cv2.putText(
                        frame,
                        "Press 0=Faust 1=Offen 2=Zeigefinger 3=OK 4=cross 5=peace 6=Thumb up",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2
                    )

                    cv2.imshow("Training Data Collector", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                    if key in GESTURES and result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]

                        # Koordinaten relativ zum Handgelenk
                        points = [
                            (lm.x, lm.y, lm.z)
                            for lm in hand.landmark
                        ]

                        wrist = points[0]
                        centered = [
                            (x - wrist[0], y - wrist[1], z - wrist[2])
                            for x, y, z in points
                        ]

                        # Skalierung anhand der größten Koordinatenspanne
                        scale = max(
                            max(p[i] for p in centered)
                            - min(p[i] for p in centered)
                            for i in range(3)
                        )

                        if scale > 1e-6:
                            features = [
                                coord / scale
                                for point in centered
                                for coord in point
                            ]

                            writer.writerow([
                                GESTURES[key], *features
                            ])
                            f.flush()

                            print(
                                f"{datetime.now().strftime('%H:%M:%S')} "
                                f"| Klasse {GESTURES[key]} gespeichert"
                            )

        self.__cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector: Data_Collector = Data_Collector(FILE)
    collector.start()