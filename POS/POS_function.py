import pickle
import jieba
import numpy as np
from keras.preprocessing import sequence

from keras.models import load_model


def pos(sentence):
    list = jieba.lcut(sentence)


    # 读取文件，标签的数组，按照索引来获取标签内容；vocab的所有单词及其index
    word_index = np.load("./x_word_index.npy", allow_pickle=True).item()

    max_dim_index_vec = tokenizer(list, word_index)


    return max_dim_index_vec


def tokenizer(words_list, word_index):
    data = []
    for word in words_list:
        try:
            data.append(word_index[word]) # 把文本中的 词语转化为index
        except:
            data.append(0)
    a = [data]
    # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    data = sequence.pad_sequences(a, maxlen=100, padding='post', truncating='post')

    return data

def predict(data):

    # 加载之前训练好的模型和标签的index
    model = load_model('./cn_pos_tag.h5')
    labels_index = np.load("./y_labels_index.npy", allow_pickle=True).item()

    print(data)
    print(data[:1])
    ans = model.predict(data[:1]).tolist()
    print(type(ans))
    print(len(ans[0][0]))

def main():
    data = pos("我们的使命就是逆战而亡")
    predict(data)

# if "__name__" == '__main__':
#     main()

main()




