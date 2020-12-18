# encoding=utf-8
import os
import codecs
import pickle
import itertools
from collections import OrderedDict
import json

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
from predict_from_doc import sentence_choose

flags = tf.app.flags
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     0,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("freeze_data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("freeze_data", "tag.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("freeze_data", "tag.test"),   "Path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)


    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def input_pro(train_data):
    if train_data:
        file_name = "golden_data.json"
    else:
        file_name = "golden_test_data.json"
    with open(file_path + file_name, "r", encoding="utf-8") as f:
        line_data = f.readlines()
        return line_data

def write_to_file(result, train_data):
    import json
    file_path = "result//"
    if train_data:
        storage_file_name = 'freeze_extract_data.json'
    else:
        storage_file_name = 'extract_test_data.json'
    with open(file_path + storage_file_name,'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

def evaluate_line(train_data):
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        import json
        line = input_pro(train_data)
        for i in range(len(line)):
            data = json.loads(line[i])
            doc = data["doc"]
            id = data["id"]
            sentences = doc.split("。")
            sentence_list = list()

            for j in range(len(sentences)):
                try:
                    sentence = sentences[j].replace(' ','')
                    result = model.evaluate_line(sess, input_from_line(sentence, char_to_id), id_to_tag)
                    # print(result)
                    sentence_list.append(result)
                except:
                    continue
            extract_data= sentence_list

            result = sentence_choose(sentence_list)
            result = json.loads(result)
            event_dict = {}
            for result_line in result:
                entities = result_line['entities']
                for entity in entities:
                    if entity['event'] != []:
                        if entity['type'] in event_dict.keys():
                            if entity['word'] not in event_dict[entity['type']]:
                                event_dict[entity['type']].append(entity['word'])
                        else:
                            event_dict[entity['type']] = [entity['word']]
            print (event_dict)
            data["result"] = sentence_list
            data["demo_result"] = event_dict
            write_to_file(data, train_data)

def test():
    make_path(FLAGS)
    config = load_config(FLAGS.config_file)
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id
    )
    test_manager  = BatchManager(test_data, 100)
    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config = tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        evaluate(sess, model, "test", test_manager, id_to_tag, logger)

def main(_):

    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        train_data = False
        evaluate_line(train_data)
        # test()

if __name__ == "__main__":
    tf.app.run(main)
