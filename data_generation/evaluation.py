import numpy
from PIL import Image
a= numpy.array(Image.open('/home/leejeyeol/Datasets/Avenue/training_videos/15/output_00118.png'),dtype=numpy.float)
a2 =Image.fromarray(a)
a2.show()
print(a)

b= numpy.array(Image.open('/home/leejeyeol/Datasets/Avenue/mean_image.png'),dtype=numpy.float)
b2 = Image.fromarray(b)
b2.show()
print(b)

c= a-b
c2 = Image.fromarray(c)
c2.show()
print(c)
d = numpy.array(numpy.round(c),numpy.uint8)
d2 = Image.fromarray(d)
d2.show()
print(d)