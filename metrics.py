import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""

    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_softmax_KL(preds1, preds2, mask):
    """Softmax cross-entropy loss with masking 实值、预测值."""
    kl = tf.keras.losses.KLDivergence()
    preds1 = tf.nn.softmax(preds1)
    preds2 = tf.nn.softmax(preds2)
    loss = kl(preds2, preds1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_L1_distance(preds1, preds2, mask):
    """Softmax cross-entropy loss with masking 实值、预测值."""
    loss = tf.norm(preds2 - preds1, ord=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)
