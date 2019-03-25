
import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    text = ""

    with open(filename, "r") as file:
        text = file.read()