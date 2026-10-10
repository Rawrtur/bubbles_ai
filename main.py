from hand_tracking.camera_predict import Gesture_prediction
from Cummunications_signal import Communication_signal
import threading
import time
from Shifter import Shifter

ANIM1: int = 0
ANIM2: int = 1

PATTERNS: list[tuple[int,int,int,int,int]] = [
    (1,1,0,0,ANIM1),
    (0,0,2,2, ANIM2)
]

class Bubbles:
    def __init__(self):
        self.__signal: Communication_signal = Communication_signal()
        self.__stop_event: threading.Event = threading.Event()
        self.__gesture_prediction: Gesture_prediction = Gesture_prediction(self.__stop_event, communication=self.__signal)
        self.__shifter: Shifter = Shifter(4)

    def predict_gesture(self)->None:
        self.__gesture_prediction.start()

    def run(self)->None:
        t1: threading.Thread = threading.Thread(target=self.predict_gesture)
        t2: threading.Thread = threading.Thread(target=self.__run)
        threads: list[threading.Thread] = [t1,t2]
        for t in threads:
            t.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Beende Programm")
        finally:
            self.__stop_event.set()
            for t in threads:
                t.join()
            print("Alle Thread beendet")

    def __run(self)->None:
        while not self.__stop_event.is_set():
            time.sleep(0.3) # im Intervall wird der Input akutalisiert
            self.__shifter.write(self.__signal.get_data())
            # print(self.__shifter)
            ## find match
            array:list = self.__shifter.get_raw_array()
            print(self.__shifter)
            for pattern in PATTERNS:
                is_pattern: bool = True
                for i in range(4):
                    if pattern[i] != array[i]:
                        is_pattern = False
                if is_pattern:
                    # print(pattern[4])
                    ...
                

if __name__ == "__main__":
    bubbles: Bubbles = Bubbles()
    bubbles.run()

