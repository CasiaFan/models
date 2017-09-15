import tensorflow as tf

slim = tf.contrib.slim

first_class_label = [0, 1, 2]
second_class_label = range(20)
start_index = [0, 8, 10, 21]


sess = tf.Session('')

first_class_onehot = slim.one_hot_encoding(first_class_label, 3)
second_class_onehot = slim.one_hot_encoding(second_class_label, 20)

logits1 = [0.5, 0.5, 0]
logits2_1 = [0.5, 0.5, 0, 0, 0, 0, 0, 0]
logits2_2 = [1., 0]
logits2_3 = [1., 0, 0, 0, 0, 0, 0, 0, 0, 0]
logits2 = [logits2_1, logits2_2, logits2_3]

end_points = {}
pred_score1 = tf.nn.softmax(logits=logits1, name="pred1")
end_points['pred1'] = pred_score1
max_pred = tf.argmax(pred_score1)
for index in range(3):
    pred_score2 = tf.nn.softmax(logits2[index], name='pred_'+str(index))
    end_points['pred2_'+str(index)] = pred_score2
print sess.run(end_points)

for index in range(3):
    tf.losses.softmax_cross_entropy(logits2[index], second_class_onehot[index][start_index[index]:start_index[index+1]], weights=tf.constant(0.5))
losses = []
for loss in tf.get_collection(tf.GraphKeys.LOSSES):
    losses.append(loss)
    print sess.run(loss)
print sess.run(tf.add_n(losses))