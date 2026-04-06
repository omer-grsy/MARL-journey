width = 5
height = 3

# TODO: Rectangle class yaz ve alan hesapla

class Rectangle:
    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError('width and height must be positive')
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height


rect= Rectangle(width, height)
print(rect.area())