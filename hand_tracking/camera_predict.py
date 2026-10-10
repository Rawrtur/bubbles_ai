from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn


# --------------------------------------------------
# 1. Pfade und Gestennamen
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH:str = BASE_DIR / "handgesten.pt"

LABELS:dict[int, str] = {
    0: "Faust",
    1: "Offene Hand",
    2: "Zeigefinger",
    3: "Ok",
    4: "cross"
}


# --------------------------------------------------
# 2. Netzwerk: MUSS zum trainierten Modell passen
# --------------------------------------------------

class HandGestureNet(nn.Module):
    def __init__(self, num_classes:int)->None:
        super().__init__()

        self.network:nn.Sequential = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# 3. Gespeichertes Modell laden
# --------------------------------------------------

class Gesture_prediction():
    def __init__(self, model_path:str)->None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modell nicht gefunden: {model_path}"
            )
        self.__checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        # Unterstützt das Speicherformat aus dem letzten train.py
        # sowie ein direkt gespeichertes state_dict.
        if "model_state_dict" in self.__checkpoint:
            state_dict:dict = self.__checkpoint["model_state_dict"]
            num_classes:int = self.__checkpoint["num_classes"]
        else:
            state_dict:dict = self.__checkpoint
            num_classes:int = state_dict["network.5.weight"].shape[0]

        self.__model: HandGestureNet = HandGestureNet(num_classes)
        self.__model.load_state_dict(state_dict)
        self.__model.eval()

        print("Modell erfolgreich geladen!")

        # 4. MediaPipe initialisieren

        self.__mp_hands = mp.solutions.hands
        self.__mp_draw = mp.solutions.drawing_utils

        self.__hands = self.__mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def __normalize_landmarks(self,hand_landmarks)->float:
        coords = np.array(
            [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
            dtype=np.float32,
        )

        # Handgelenk als Ursprung
        coords -= coords[0].copy()

        # Skalierung unabhängig von der Handgröße
        scale = max(
            float(np.ptp(coords[:, 0])),
            float(np.ptp(coords[:, 1])),
            float(np.ptp(coords[:, 2])),
        )

        if scale < 1e-6:
            return None

        coords /= scale

        return coords.flatten()


# --------------------------------------------------
# 6. Webcam starten
# --------------------------------------------------
    def start(self)->None:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.__hands.close()
            raise RuntimeError(
                "Webcam konnte nicht geöffnet werden."
            )

        try:
            with torch.no_grad():
                while True:
                    success, frame = cap.read()

                    if not success:
                        print("Kein Kamerabild erhalten.")
                        break

                    # Spiegelbild wie bei der Datensammlung
                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    results = self.__hands.process(rgb)

                    prediction_text: str = "Keine Hand erkannt"

                    if results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]

                        # Handpunkte einzeichnen
                        self.__mp_draw.draw_landmarks(
                            frame,
                            hand,
                            self.__mp_hands.HAND_CONNECTIONS,
                        )

                        features = self.__normalize_landmarks(hand)

                        if features is not None:
                            input_tensor = torch.from_numpy(
                                features
                            ).unsqueeze(0)

                            logits = self.__model(input_tensor)
                            probabilities = torch.softmax(
                                logits, dim=1
                            )

                            confidence, predicted = (
                                probabilities.max(dim=1)
                            )

                            class_id = predicted.item()
                            confidence_value = confidence.item()

                            gesture_name = LABELS.get(
                                class_id,
                                f"Klasse {class_id}",
                            )

                            prediction_text = (
                                f"{gesture_name} "
                                f"({confidence_value:.0%})"
                            )

                    # Ergebnis oben im Kamerabild anzeigen
                    cv2.putText(
                        frame,
                        prediction_text,
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(
                        "Meine Handgestenerkennung",
                        frame,
                    )

                    # Mit Q beenden
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            self.__hands.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    GP: Gesture_prediction = Gesture_prediction(MODEL_PATH)
    GP.start()