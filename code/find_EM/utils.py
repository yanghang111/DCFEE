import codecs, json, logging, shutil
import os, re, math, random
import numpy as np
import tensorflow as tf

def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embedding.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with'
          'pretrained embeddings.'% (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights

def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.skpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def input_from_line(str_token, char_to_id, train=False):
    chars = [char_to_id[w if w in char_to_id else '<UNK>']
             for w in str_token]
    if train:
        tags = 0
    else:
        tags = 0
    data = [[str_token], [chars], [tags]]
    return data

def create_dico(item_list):
    """
    根据词频创建词典
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    从词频字典创建映射（word到ID / ID到word）。按照频率递减排序。
    :param dico: 词频词典
    :return: 词和ID的一一对应字典
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i,v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def load_sentence(path):
    """
    :param path:  load_path
    :return:
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = json.loads(line)
        doc_id = line[0]
        sentence_text = line[1]
        tag = line[-1]
        sentence.append(sentence_text)
        sentences.append(line)
    chars = [[x for x in s] for s in sentence]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    return sentences, dico, char_to_id, id_to_char

def create_tag_dico(item_list):
    """
    根据词频创建词典
    """
    assert type(item_list) is list
    dico = {}
    for item in item_list:
        if item not in dico:
            dico[item] = 1
        else:
            dico[item] += 1
    return dico

def tag_mapping(sentences):
    """
    创建tags的映射字典，频率排序
    :param sentences:
    :return:
    """
    # for s in sentences:
    #     print(s[-1])
    tags = [s[-1] for s in sentences]
    dico = create_tag_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, train=True):
    train_data = []
    test_data = []
    dev_data = []
    positive_sample_num = 0
    nagtive_sample_num = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        id = sentence[0]
        chars = [char_to_id[w if w in char_to_id else '<UNK>'] for w in sentence[1]]
        tags = int(sentence[-1])
        if id.split("-")[0] == "DEV":
            if sentence[-1] == "1":
                nagtive_sample_num += 1
                train_data.append([sentence, chars,tags])
            elif sentence[-1] != "1":
                positive_sample_num += 1
                train_data.append([sentence, chars, tags])
        elif id.split("-")[0] == "TST3" or id.split("-")[0] == "TST4":
            test_data.append([sentence, chars,tags])
        else:
            dev_data.append([sentence, chars,tags])
    print("正负样本比例：",positive_sample_num/nagtive_sample_num)
    return train_data, test_data, dev_data

def create_model(session, Moeld_class, path, load_vec, config, id_to_char, logger):
    # create model , reuse parameters if exists
    model = Moeld_class(config)
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = session.run(model.char_lookup.read_value())
            emb_weights = load_vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model

def augment_with_pretrained(dictionary,ext_emb_path):
    """
    用预先训练的词向量增加字典。
    :param dictionary:词频字典
    :param ext_emb_path:预训练好的词向量
    :param chars: 测试集的词
    :return:
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)
    # 从文本中加载词向量
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
    ])
    for char in pretrained:
        if char not in dictionary:
            dictionary[char] = 0

    word_to_id,id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def make_path(params):
    """
    为训练过程和结果创建文件夹
    """
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir("log"):
        os.makedirs("log")

def load_config(config_file):
    """
    加载模型的参数配置,参数以json格式存储
    """
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)

def save_config(config, config_file):
    """
    存储模型的参数配置，并以json格式存储
    """
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f , ensure_ascii=False, indent=4)

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def print_config(config, logger):
    """
    输出模型的参数配置
    """
    with open("result.json", 'a', encoding="utf-8") as outfile:
        for k, v in config.items():
            logger.info("{}:\t{}".format(k.ljust(15), v))
            json.dump("{}:{}".format(k.ljust(15), v), outfile, ensure_ascii=False)
            outfile.write('\n')

def test_ner(results, path):
    output_file = os.path.join(path, "predict.utf8")
    with open(output_file, "w", encoding='utf-8') as f:
        to_write = []
        for line in results:
            to_write.append(line + "\n")
        f.writelines(to_write)

class BatchManager(object):

    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[1]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        entity_words = []
        targets = []
        max_length = max([len(sentence[1]) for sentence in data])

        for line in data:
            sentence, char, tags = line
            padding = [0] * (max_length - len(char))
            strings.append(list(sentence[0]) + padding)
            chars.append(char + padding)
            targets.append(tags)
        return [strings, chars, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]