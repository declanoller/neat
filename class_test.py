


class myCoolClass:

    def __init__(self, val):

        self.v = val




def testFn(c):


    print('before:', c.v)
    c.v = 5
    print('after:', c.v)




d = myCoolClass(8)
print('before fn:', d.v)
testFn(d)

print('outside fn:', d.v)








#
