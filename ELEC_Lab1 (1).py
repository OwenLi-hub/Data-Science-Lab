import numpy as np
#Question 1
list_1 = [-1, 2, 3, 9, 0]
list_2 = [1, 2, 7, 10, 14]
print(list_1 + list_2)

#Question 2
list_1 = [-1, 2, 3, 9, 0]
list_2 = [1, 2, 7, 10, 14]
my_list = []

for i in range (0, 4):
    my_list.append(list_1[i] + list_2[i])
print(my_list)

#Question 3
my_4d_array = np.array([[[[0.5], [1]], [[1.5], [2]], [[2.5], [3]]], [[[3.5], [4]], [[4.5], [5]], [[5.5], [6]]]])
print(my_4d_array.shape)

#Question 4
my_3d_array = np.arange(27).reshape(3, 3, 3)
print(f"{my_3d_array[:, :, 0]} \n")
print(f"{my_3d_array[1, 1, :]} \n")
print(f"{my_3d_array[:, 0::2, 0::2]} \n")

#Question 5
my_3d_array = np.arange(27).reshape(3, 3, 3)
print(my_3d_array[[0, 1, 2], [1, 2, 0], [1, 2, 0]])
print(my_3d_array[[1], [0, 2], [0, 2]])

#Question 6
my_2d_array = np.arange(-10, 20).reshape(5, 6)
sum_of_rows = my_2d_array.sum(axis=0)
indexing_array = sum_of_rows % 10 == 0
print(my_2d_array[:, indexing_array])