# encoding:utf-8

# @Author: Rilzob
# @Time: 2018/11/6 上午10:16
import config
import utils
from torch.autograd import Variable
import torch
from torch import optim
import models


def main(args):
    # preprocessing: word(en, cn) -> number(one hot vector)

    # load sentences (English and Chinese)
    train_en, train_cn = utils.load_data(args.train_file)
    dev_en, dev_cn = utils.load_data(args.dev_file)
    args.num_train = len(train_en)
    args.num_dev = len(dev_en)

    en_dict, en_total_words = utils.build_dict(train_en)
    cn_dict, cn_total_words = utils.build_dict(train_cn)
    inv_en_dict = {v: k for k, v in en_dict.items()}
    inv_cn_dict = {v: k for k, v in cn_dict.items()}

    args.en_total_words = en_total_words
    args.cn_total_words = cn_total_words

    # encode the words into numbers
    train_en, train_cn = utils.encode(train_en, train_cn, en_dict, cn_dict)
    dev_en, dev_cn = utils.encode(dev_en, dev_cn, en_dict, cn_dict)

    # convert the train and dev data into numpy matrices
    # batch_size * seq_length
    train_data = utils.gen_examples(train_en, train_cn, args.batch_size)
    dev_data = utils.gen_examples(dev_en, dev_cn, args.batch_size)

    model = models.EncoderDecoderModel()

    crit = utils.LanguageModelCriterion()
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameter(), lr=learning_rate)

    for epoch in range(args.num_epochs):
        for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in enumerate(train_data):
            # convert numpy ndarray to Pytorch tensor
            # convert to Pytorch Variable
            batch_size = mb_x.shape[0]

            mb_x = Variable(torch.from_numpy(mb_x)).long()
            mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
            hidden = model.init_hidden(batch_size)
            mb_input = Variable(torch.from_numpy(mb_y[:, :-1])).long()
            mb_out = Variable(torch.from_numpy(mb_y[:, 1:])).long()
            mb_out_mask = Variable(torch.from_numpy(mb_y_mask[:, 1:])).long()

            mb_pred, hidden = model(mb_x, mb_x_mask, mb_input, hidden)

            # calculate loss function
            loss = crit(mb_pred, mb_out, mb_out_mask)

            # update the model
            optimizer.zero_grad()  # zero the previous gradient
            loss.backward()  # calculate gradient
            optimizer.step()  # gradient descent
            # compare mb_out with mb_pred


if __name__ == '__main__':
    args = config.get_args()
    main(args)