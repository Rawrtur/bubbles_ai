import threading
import time

def aufgabe1():
    for i in range(5):
        print(f"Thread 1: {i}")
        time.sleep(1)

def aufgabe2():
    for i in range(5):
        print(f"Thread 2: {i}")
        time.sleep(1)

# Threads erstellen
t1 = threading.Thread(target=aufgabe1)
t2 = threading.Thread(target=aufgabe2)

# Beide Threads starten
t1.start()
t2.start()

# Auf das Ende beider Threads warten
t1.join()
t2.join()

print("Beide Threads sind fertig.")