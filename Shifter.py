class Shifter:
    def __init__(self, size:int)->None:
        self.__array: list = [None] * size
        self.__size: int = size

    def write(self, item)->None:
        self.__array.append(item)
        if len(self.__array) > self.__size:
            self.__array = self.__array[1:]

    def __str__(self)->str:
        return str(self.__array)

    def get_raw_array(self)->list:
        return self.__array

if __name__ == "__main__":
    shifter: Shifter = Shifter(2)
    shifter.write(1)
    shifter.write(2)
    shifter.write(3)
    shifter.write(4)
    print(shifter)