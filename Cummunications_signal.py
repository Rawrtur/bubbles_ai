class Communication_signal():
    def __init__(self)->None:
        self.__data = None

    def set_data(self, data)->None:
        self.__data = data

    def get_data(self):
        return self.__data