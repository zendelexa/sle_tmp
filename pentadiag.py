from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

N = 5
M = 10


def calc(alpha, beta, gamma=0):
    A = [[0 for j in range(N)] for i in range(N)]
    for i in range(N):
        A[i][i] = alpha
        if i > 0:
            A[i][i - 1] = beta
        if i < N - 1:
            A[i][i + 1] = beta
    A = np.array(A)
    # print("DET:", np.linalg.det(A@A))
    # print("VAL:", np.linalg.det(A) / (beta ** N))

    B = [[0 for j in range(N)] for i in range(N)]
    for i in range(N):
        B[i][i] = beta
        if i > 0:
            B[i][i - 1] = gamma
        if i < N - 1:
            B[i][i + 1] = gamma
    B = np.array(B)
    # print(B)
    invB = np.linalg.inv(B)
    # print(invB)

    # print("aaa:", np.linalg.det(A @ A - 4 * B @ B))
    print("VAL 2:", (np.linalg.det(A) + (abs(np.linalg.det(A @ A - 4 * B @ B))) ** 0.5) / (2 * beta ** N))
    # print("VAL 3:",
    #       (np.linalg.det(A) + (abs(np.linalg.det(A @ A)) + abs(np.linalg.det(4 * B @ B))) ** 0.5) / (2 * beta ** N))

    x = np.array([1 for i in range(N)])

    xi = x
    print(0, xi)
    for step in range(10):
        xp = np.zeros(N)
        # xp = np.array([1 for _ in range(N)])
        for i in range(1, M):
            axi = A @ xi
            bxpaxi = B @ xp + A @ xi
            xp, xi = xi, invB @ (-B @ xp - A @ xi)
            print(i, xi)
    return xi


def calc_norm(beta, gamma):
    return np.linalg.norm(calc(beta, gamma))


# print(calc(13, -10, gamma=8))  # При N = 5
# print(calc(1, 4.5, gamma=0))
# print(calc(1, 4.5, gamma=0))
#
# for i in np.linspace(1.1, 10, 100):
#     print(i, end=":\t")
#     calc(1, i)
print(calc(1, 3.71))

# print(20**0.5)
#
# print(calc(16))
#
# x, y = np.meshgrid(np.linspace(5, 10, 10), np.linspace(0.5, 0.7, 10))
# z = np.vectorize(calc_norm)(x, y)
#
# fig = plt.figure(figsize=plt.figaspect(0.5))  # Создаем фигуру
# ax = fig.add_subplot(111, projection='3d')  # 3D-оси
#
# # Отображение точек
# ax.plot_surface(x, y, z, cmap="viridis", label='Точки')
#
# # Подписи осей
# ax.set_xlabel('Ось X')
# ax.set_ylabel('Ось Y')
# ax.set_zlabel('Ось Z')
# #ax.set_zscale('log')
#
# # Заголовок
# ax.set_title('3D-график рассеяния точек')
#
# # Легенда
# ax.legend()
#
# plt.show()

# x = np.linspace(-20, -10, 30)
# y = np.vectorize(calc_norm)(x)

# plt.figure(figsize=(8, 5))  # Размер графика
# plt.plot(x, y, 'b-', linewidth=2)  # Синяя линия
#
# # Подписи
# plt.xlabel('Ось X')
# plt.ylabel('Ось Y')
# plt.yscale('log')
# plt.grid(True)  # Сетка
#
# plt.show()

