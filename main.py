import numpy as np

def log(index, arg):
    print(index, end=" → ")
    print(arg)
    print('--------------------------------------------------------')

# 1.
log(1, np.zeros(15))

# 2.
log(2, np.full(8, 3.2))

# 3.
test3 = np.zeros(15)
test3[4] = 1
log(3, test3)

# 4.
log(4, np.arange(12, 44))

# 5.
log(5, np.random.random((3, 3, 2)))

# 6.
test6 = np.random.random((10, 10))
test6min, test6max = test6.min(), test6.max()
log(6, [test6min, test6max])

# 7. ???
test7 = np.ones((8, 8))
test7[1:-1, 1:-1] = 0
log(7, test7)

# 8.
test8 = np.zeros((8, 8), dtype=int)
test8[1::2, ::2] = 1
test8[::2, 1::2] = 1
log(8, test8)

# 9.
log(9, np.tile(np.array([[0, 1], [1, 0]]), (4, 4)))

# 10.
log(10, np.dot(np.ones((4, 2)), np.ones((2, 5))))

# 11.
test11 = np.arange(11)
test11[(4 < test11) & (test11 < 7)] *= -1
log(11, test11)

# 12.
test12 = np.zeros((6, 6))
test12 += np.arange(6)
log(12, test12)

# 13.
test13 = np.random.random(13)
test13.sort()
log(13, test13)

# 14.
arr1 = np.random.randint(0, 6, 12)
arr2 = np.random.randint(0, 6, 12)
test14 = np.allclose(arr1, arr2)
log(14, test14)

# 15.
test15 = np.random.random(15)
test15[test15 == test15.max()] = -1

# Alternative method
# test15[test15.argmax()] = -1
log(15, test15)

# 16.
test16 = np.arange(70)
value = np.random.uniform(0, 70)
index = (np.abs(test16 - value)).argmin()
log(16, test16[index])

# 17. ???
test17 = np.zeros(10, [
    ('position', [
        ('x', float, 1),
        ('y', float, 1)
    ]),
    ('color', [
        ('r', float, 1),
        ('g', float, 1),
        ('b', float, 1)
    ])
])
log(17, test17)

# 18.
val = np.random.rand(3, 6)
test18 = val - val.mean(axis=1, keepdims=True)
log(18, test18)

# 19.
test19 = np.arange(36).reshape(6, 6)
test19[[1, 0]] = test19[[0, 1]]
log(19, test19)

# 20. Рассмотрим 2 набора точек P0, P1 описания линии (2D) и точку р, как вычислить
# расстояние от р до каждой линии i (P0[i],P1[i]). Поиск расстояния от токи p до линии i
# (P0[i],P1[i]) задать отдельной функцией. ???
def distance(P0, P1, p):
    T = P1 - P0
    L = (T ** 2).sum(axis=1)
    U = -((P0[:, 0] - p[..., 0]) * T[:, 0] + (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U * T - p
    return np.sqrt((D ** 2).sum(axis=1))

P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10, (10, 2))
p  = np.random.uniform(-10, 10, (1, 2))
log(20, distance(P0, P1, p))

# 21.
test21 = np.arange(33)
np.random.shuffle(test21)
number = 3
log(21, test21[np.argpartition(-test21, number)[:number]])

# 22. ???

# 23. ???

# 24. ???

# 25. ???