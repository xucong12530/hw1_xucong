import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv

class network_xc():
    def __init__(self, lr=0.001, num_in=100, num_out=10, hidden=[3, 4], weight_scale=1e-3, batch_size = 200,
                 L2=0.5, epoch=200):
        self.lr = lr
        self.num_in = num_in
        self.num_out = num_out
        self.params = {}
        self.hidden = hidden
        self.weight_scale = weight_scale
        self.L2 = L2
        self.epoch = epoch
        self.batch_size = batch_size

        self.loss = []
        self.acc = []

        self.flag_init_weight = False
        if self.flag_init_weight == False:
            self.init_weights()

    def init_weights(self):
        assert self.flag_init_weight == False
        self.params['W1'] = np.random.randn(self.num_in, self.hidden[0]) * self.weight_scale
        self.params['W2'] = np.random.randn(self.hidden[0], self.hidden[1]) * self.weight_scale
        self.params['W3'] = np.random.randn(self.hidden[1], self.num_out) * self.weight_scale
        self.params['b1'] = np.zeros(self.hidden[0], )
        self.params['b2'] = np.zeros(self.hidden[1], )
        self.params['b3'] = np.zeros(self.num_out, )
        self.flag_init_weight = True

    def loss_softmax(self, x, y):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

    def train(self, input, y):
        for i in range(self.epoch):
            loss = 0
            for j in range(0,input.shape[0],self.batch_size):
                # forward
                tmp_input = input[j:j+self.batch_size,]
                tmp_y = y[j:j+self.batch_size,]
                hidden1_ = np.dot(tmp_input, self.params['W1']) + self.params['b1']
                hidden1 = 1 / (np.exp(-1 * hidden1_) + 1)

                hidden2_ = np.dot(hidden1, self.params['W2']) + self.params['b2']
                hidden2 = 1 / (np.exp(-1 * hidden2_) + 1)

                output = np.dot(hidden2, self.params['W3']) + self.params['b3']
                if j == 0:
                    output_tmp = output
                else:
                    output_tmp = np.append(output_tmp, output)

                tmp_loss, dout = self.loss_softmax(output, tmp_y)
                loss += 0.5 * self.L2 * (
                        np.sum(np.square(self.params['W3'])) + np.sum(np.square(self.params['W2'])) + np.sum(
                    np.square(self.params['W1']))) + tmp_loss

                dW3 = np.dot(hidden2.T, dout)
                db3 = np.sum(dout, axis=0)
                dhidden2 = np.dot(dout, self.params['W3'].T) * (1 - hidden2) * hidden2

                dW2 = np.dot(hidden1.T, dhidden2)
                db2 = np.sum(dhidden2, axis=0)
                dhidden1 = np.dot(dhidden2, self.params['W2'].T) * (1 - hidden1) * hidden1

                dW1 = np.dot(tmp_input.T, dhidden1)

                db1 = np.sum(dhidden1, axis=0)


                dW3 += self.params['W3'] * self.L2
                dW2 += self.params['W2'] * self.L2
                dW1 += self.params['W1'] * self.L2

                # backward
                self.params['W3'] -= self.lr * dW3
                self.params['b3'] -= self.lr * db3
                self.params['W2'] -= self.lr * dW2
                self.params['b2'] -= self.lr * db2
                self.params['W1'] -= self.lr * dW1
                self.params['b1'] -= self.lr * db1


            self.loss.append(loss)

            output_tmp = output_tmp.reshape(input.shape[0],10)

            y_pred = np.argmax(output_tmp, axis=1).reshape(1, -1)
            y_true = y.reshape(1, -1)
            sum_ = 0.0
            for c in range(y_pred.shape[1]):
                if y_pred[0, c] == y_true[0, c]:
                    sum_ = sum_ + 1
                    acc = 100.0 * sum_ / y_pred.shape[1]
                    self.acc.append(acc)

            if i % 10 == 0:
                print('Epochs {} -- Acc: [{:.3f}%], Loss: [{:.5f}]'.format(i, 100.0 * sum_ / y_pred.shape[1], loss))

            if i < 500:
                self.lr = self.lr
            elif i < 1000 and i >= 500:
                self.lr -= self.lr * 0.6
            else:
                self.lr = self.lr * 0.1

            if self.lr < 0.1:
               self.lr = 0.1

            if i == self.epoch - 1:
                y_pred = np.argmax(output_tmp, axis=1).reshape(1, -1)
                y_true = y.reshape(1, -1)
                sum_ = 0.0
                for c in range(y_pred.shape[1]):
                    if y_pred[0, c] == y_true[0, c]:
                        sum_ = sum_ + 1
                print('Epochs {} -- Acc: [{:.3f}%], Loss: [{:.5f}]'.format(i, 100.0 * sum_ / y_pred.shape[1], loss))

    def test(self, input, y):
        hidden1_ = np.dot(input, self.params['W1']) + self.params['b1']
        hidden1 = 1 / (np.exp(-1 * hidden1_) + 1)

        hidden2_ = np.dot(hidden1, self.params['W2']) + self.params['b2']
        hidden2 = 1 / (np.exp(-1 * hidden2_) + 1)

        output_ = np.dot(hidden2, self.params['W3']) + self.params['b3']
        output = 1 / (np.exp(-1 * output_) + 1)

        y_pred = np.argmax(output, axis=1).reshape(1, -1)
        y_true = y.reshape(1, -1)
        sum_ = 0.0
        for c in range(y_pred.shape[1]):
            if y_pred[0, c] == y_true[0, c]:
                sum_ = sum_ + 1
        print('Test acc is {:.5f}'.format(sum_ / y_pred.shape[1]))
        return sum_ / y_pred.shape[1]

    def get_loss_history(self):
        return self.loss

    def get_acc_history(self):
        return self.acc

    def save_para(self):
        np.save('W1.npy',self.params['W1'])
        np.save('W2.npy', self.params['W2'])
        np.save('W3.npy', self.params['W3'])
        np.save('b1.npy', self.params['b1'])
        np.save('b2.npy', self.params['b2'])
        np.save('b3.npy', self.params['b3'])

train = csv.reader(open('mnist_train.csv', 'r'))
train_content = []
for line in train:
    train_content.append(line)

test = csv.reader(open('mnist_test.csv', 'r'))
test_content = []
for line in test:
    test_content.append(line)
train_content = np.array(train_content, dtype=np.float32)
test_content = np.array(test_content, dtype=np.float32)

train_label = np.array(train_content[:, 0], dtype=np.int)
train_x = train_content[:,1 :]
test_label = np.array(test_content[:, 0], dtype=np.int)
test_x = test_content[:, 1:]

assert train_x.shape[1] == test_x.shape[1]
print('Number of input is %d' % train_x.shape[1])
num_input = train_x.shape[1]

train_x = (train_x - 255/2) / 255
test_x = (test_x - 255/2) / 255


lr = 0.8
hidden=[50, 40]
weight=0.1
batch=1000
L2=5e-4
epoch=2000

model = network_xc(lr=lr, num_in=784, num_out=10, hidden=hidden, weight_scale=weight,batch_size=batch, L2=L2, epoch=epoch)
model.train(train_x, train_label)
model.test(test_x, test_label)

model.save_para()

loss = model.get_loss_history()
plt.plot(loss)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

acc = model.get_acc_history()
plt.plot(acc)
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

