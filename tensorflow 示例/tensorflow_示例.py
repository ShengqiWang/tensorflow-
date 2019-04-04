import tensorflow as tf
import numpy as np
import scipy.io
def add_layer(inputs, in_size, out_size, w_n,b_n,activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]),name=w_n)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name=b_n)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 1.训练的数据
# Make up some real data 
#x_data = np.linspace(-1,1,300)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise



data1 = scipy.io.loadmat('D:/train.mat')  
x=data1['x']
y=data1['y']
z=data1['z']
x_data=np.hstack((x,y))
y_data=z
data1 = scipy.io.loadmat('D:/test.mat')  
x=data1['x']
y=data1['y']
z=data1['z']
x_data2=np.hstack((x,y))
y_data2=z


# 2.定义节点准备接收数据
# define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
l1 = add_layer(xs, 2, 10, 'w1','b1',activation_function=tf.tanh)
l2 = add_layer(l1, 10, 15, 'w2','b2',activation_function=tf.tanh)
l3 = add_layer(l2, 15,20,'w3','b3',activation_function=tf.tanh)
l4= add_layer(l3, 20, 10, 'w4','b4',activation_function=tf.tanh)
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l4, 10, 1, 'w5','b5',activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data    
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction)),
                     #reduction_indices=[1])
loss = tf.reduce_mean(tf.square(ys - prediction))
# 5.选择 optimizer 使 loss 达到最小                   
# 这一行定义了用什么方式去减少 loss，学习率是 0.1       
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
test=1-tf.reduce_mean(tf.abs(tf.round(prediction)-ys))

# important step 对所有变量进行初始化
init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(200000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 200 == 0:
        # to see the step improvement
        print(sess.run(test, feed_dict={xs: x_data2, ys: y_data2}))
print('saving model.....')
saver=tf.train.Saver()
saver=tf.train.Saver(max_to_keep=1)
saver.save(sess,'c/mnist',global_step=0)
reader = tf.train.NewCheckpointReader('c/mnist-0')
all_variables = reader.get_variable_to_shape_map()
dataNew = 'D://outcome.mat'
scipy.io.savemat(dataNew, {'b1': reader.get_tensor('b1'),'w1': reader.get_tensor('w1'),'b2': reader.get_tensor('b2'),'w2': reader.get_tensor('w2'),'b3': reader.get_tensor('b3'),'w3': reader.get_tensor('w3'),'b4': reader.get_tensor('b4'),'w4': reader.get_tensor('w4'),'b5': reader.get_tensor('b5'),'w5': reader.get_tensor('w5')})
print('finish!')