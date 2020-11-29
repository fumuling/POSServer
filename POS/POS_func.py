# coding='UTF-8'
###
# 将训练好的词性标注模型封装为服务
###
# 加载存储的模型开始预测
from keras.preprocessing import sequence
from keras.models import load_model
import numpy as np
import jieba


# 由于将词语转化为索引的word_index需要与词向量模型对齐，故在导入词向量模型后再将X进行处理
def tokenizer(texts, word_index):
    data = []
    MAX_SEQUENCE_LENGTH = 100
    for sentence in texts:
        new_sentence = []
        for word in sentence:
            try:
                new_sentence.append(word_index[word])  # 把文本中的 词语转化为index
            except:
                new_sentence.append(0)
            
        data.append(new_sentence)
    # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    data = sequence.pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    return data


def transfer_input(input_text_segged):
    # 序号化 文本，tokenizer句子，并返回每个句子所对应的词语索引，并填充句子长度至100
    # 因为只有一句，所以在该输入句外加上了括号
    text_dictionary = np.load('D:\project\POSServer\POS/x_word_index.npy', allow_pickle=True).item()
    padded_input_text_indexs = tokenizer([input_text_segged], text_dictionary)
    
    return padded_input_text_indexs


def transfer_output(output):
    # 将输出转化为标签格式
    label_dictionary = np.load('D:\project\POSServer\POS/y_labels_index.npy', allow_pickle=True).item()
    label_dictionary_index_label = {label_dictionary[key]:key for key in label_dictionary}
    
    # 将预测的每一个标签对应向量转化为索引号
    output_index = []
    for label_vector in output[0]:
        label_max_prob = max(label_vector.tolist())
        label_max_prob_index = label_vector.tolist().index(label_max_prob)
        output_index.append(label_max_prob_index)
    processed_output = [label_dictionary_index_label[index] for index in output_index]

    return processed_output


def pos_text_seg(text_to_be_seg):
    # 将输入文本分词
    print("Processing input text...")
    segged_text = jieba.cut(text_to_be_seg, cut_all=False)
    segged_text = '/'.join(segged_text)
    segged_text = segged_text.split('/')
    print('Segged text：  ', segged_text)

    return segged_text


def pos_text_predict(segged_text):
    # 将分词后的输入文本对应模型训练时所用的词典，转化为index
    processed_input = transfer_input(segged_text)
    
    # 模型预测
    print("Using loaded model to predict...")
    pos_model = load_model("D:\project\POSServer\POS/cn_pos_tag.h5")
    output = pos_model.predict(processed_input)
    
    # 将输出转化为标签格式
    processed_output = transfer_output(output)
    
    # 将原词和词性组合输出
    final_output = []
    for index in range(len(segged_text)):
        final_output.append({segged_text[index]:processed_output[index]})

    return final_output

def get_list(final_output):
    key_list = []
    value_list = []
    for i in final_output:
        for key, value in i.items():
            key_list.append(key)
            value_list.append(value)

    return (key_list, value_list)




if __name__ == '__main__':
    # 导入训练好的词性标注深度学习模型
    print("Loading model...")
    pos_model = load_model("./cn_pos_tag.h5")
    print("Model loaded!")
    
    while True:
        # 接收输入文本
        print('\n', '*'*20, 'POS start', '*'*20)
        text_to_be_tag = input("\nPlease input your sentence(if you want to quit, just type in quit):\n")
        if text_to_be_tag == 'quit':
            break
        print('\n', '*'*20, 'Sentence processing...', '*'*20)
        segged_text = pos_text_seg(text_to_be_tag)
        print('\n', '*'*20, 'POS output', '*'*20)
        final_output = pos_text_predict(segged_text)
        a, b = get_list(final_output)


