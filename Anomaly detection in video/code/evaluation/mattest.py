import scipy.io
from matplotlib import pyplot as plt
mat = scipy.io.loadmat('/Avenue Dataset/testing_label_mask/8_label.mat')

for key in mat.keys():
    print(key)
print(type(mat.get('volLabel')))
print(mat.get('volLabel').shape)
print(mat.get('volLabel')[0].shape)
vol = mat.get('volLabel')
#
print(vol[0,0])
# for array in vol[0,0]:
#     for elem in array:
#         if elem == 1:
#             print("hurray")
#         else:
#             if elem == 0:
#                 print("boo")
#             else:
#                 print("ohh shit")
print(vol[0,34].shape)
plt.imshow(vol[0,8],cmap="gray")
plt.show()