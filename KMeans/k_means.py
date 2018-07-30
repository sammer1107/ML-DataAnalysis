import tensorflow as tf
from KMeans import KMeansExperimental
import numpy as np
import matplotlib.pyplot as plt

a = np.zeros([4,4])
b = [1,2,3,0]
a[np.arange(4),b] = 1

#o_input = np.array([[2,1],[2,2],[2,3],[3,2],[1,2],[7,6],[7,7],[7,8],[8,7],[6,7],[5,5]])
o_input = np.concatenate([np.random.normal(3,1.,[800,2]),
                          np.random.normal(1,0.1,[300,2])])
#o_input = o_input/np.max(o_input)

inp = tf.Variable(o_input,
                  trainable=False,
                  dtype=tf.float32)
k_means = KMeansExperimental(inputs=inp,
                             num_clusters=2,
                             initial_clusters="kmeans_plus_plus",
                             distance_metric='squared_euclidean',
                             experimental_score=False)

all_scores, cluster_idx, scores, cluster_initialized, init_op, train_op = k_means.training_graph()

train_iter = 30

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    initialized, _ = sess.run([cluster_initialized, init_op])
    print("ran initialize op : initialized : {} ".format(initialized))
    while not initialized:
        initialized, _ = sess.run([cluster_initialized, init_op])
        print("ran initialize op : initialized : {} ".format(initialized))
    plt.scatter(o_input[:,0], o_input[:,1])
    for i in range(train_iter):
        # nb_out, distances_out, scores_out, idx, centers = sess.run(["nearest_neighbors:0", "max_distances:0", scores, cluster_idx, train_op])
        idx, centers = sess.run([cluster_idx, train_op])
        print(centers)

        # print(np.concatenate([np.reshape(scores_out,[-1,1]), np.reshape(nb_out, [-1,1])], axis=1))
        # # calculate middle line
        centers = np.array(centers)
        m = -(centers[0][0] - centers[1][0])/(centers[0][1] - centers[1][1])
        center_of_centers = (centers[0] + centers[1])/2
        middle_line_x = [center_of_centers[0]-10,center_of_centers[0]+10]
        middle_line_y = [center_of_centers[1]-10*m,center_of_centers[1]+10*m]
        # map color
        color = np.zeros([len(idx[0]),3])
        color[np.arange(len(idx[0])),idx[0]] = 1

        # print("train_op {}".format(i))
        plt.axis([0, 5, 0, 5])
        plt.scatter(o_input[:,0], o_input[:,1], c=color, s=5)
        plt.scatter(centers[:,0], centers[:,1], c="k", marker="o", s=30)
        plt.plot(middle_line_x, middle_line_y)

        if i < train_iter-1:
            plt.draw()
            plt.pause(0.33)
            plt.clf()
        else:
            print('done')
            plt.show()



