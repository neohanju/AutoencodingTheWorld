#todo : just for test ===================
import numpy
from PIL import Image
import torch
'''
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
'''

'''
for i in range(0, 4259):
    testfalse = torch.load("/mnt/fastdataset/centering_test_false/%05d.t7" % i)
    testtrue = torch.load("/mnt/fastdataset/centering_test_false/%05d.t7" % i)
    print(i," : ",sum(sum(sum(testfalse-testtrue))))
'''
#=========================================