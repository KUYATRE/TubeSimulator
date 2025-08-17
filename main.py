import tensorflow as tf

x = [1, 2, 3, 4, 5]
y = [-1, -2, -3, -4, -5]

# 여기에 코드를 작성하세요.
dataset = tf.data.Dataset.from_tensor_slices((x, y))

for data in dataset:
    print(f"{data[0].numpy()}, {data[1].numpy()}")