import tensorflow as tf
import os, pickle
from utils import load_sentence, augment_with_pretrained, tag_mapping, prepare_dataset, test_ner
from utils import BatchManager, make_path, load_config, save_config, get_logger, print_config, create_model, load_word2vec, save_model, input_from_line
from model import Model

from utils import clean
import os, json, shutil, logging

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,          "clean train floder")
flags.DEFINE_boolean("train",       False,          "Wither train the model")
# configuerations for the model

flags.DEFINE_integer("num_steps",   50,              "num_steps")
flags.DEFINE_integer("char_dim",    100,             "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,             "Num of hidden units in LSTM")

# configuration for training
flags.DEFINE_float("clip",          5,                  "Gradient clip")
flags.DEFINE_float("dropout",       0.5,                "Dropout rate")
flags.DEFINE_integer("batch_size",    50,                 "batch size")
flags.DEFINE_float("lr",            0.001,               "Initiaal learning rate")
flags.DEFINE_string("optimizer",    "adam",              "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,               "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,              "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       False,               "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,                "maximum trainning epochs")
flags.DEFINE_integer("steps_check", 100,                "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",              "Path to save model")
flags.DEFINE_string("summary_path", "summary",          "Path to store summaries")
flags.DEFINE_string("log_file",     "train_log",        "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",         "File for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",       "File for vocab")
flags.DEFINE_string("config_file",  "config_file",      "File for config")
flags.DEFINE_string("script",       "conlleval",        "evaluation scropt")
flags.DEFINE_string("result_path",  "result",           "Path for results")
flags.DEFINE_string("emb_file",     "100.utf8",    "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "muc_mention.json"),   "path for train data")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip <5.1,         "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1,  "dropout rate should between 0 and 1"
assert FLAGS.lr > 0,            "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

def config_model(char_to_id):
    config = dict()
    config["num_char"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config["num_steps"] = FLAGS.num_steps

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def evaluate(sess, model, name, data,logger):
    logger.info("evaluate:{}".format(name))
    results, acc = model.evaluate(sess, data)
    return acc

def train():
    train_sentences, dico, char_to_id, id_to_char = load_sentence(FLAGS.train_file)
    if not os.path.isfile(FLAGS.map_file):
        if FLAGS.pre_emb:
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico.copy(),
                FLAGS.emb_file,
                )
        else:
            sentences, dico, char_to_id, id_to_char = load_sentence(FLAGS.train_file)
        print(train_sentences[0])
        with open(FLAGS.map_file, 'wb') as f:
            pickle.dump([char_to_id, id_to_char], f)
    else:
        with open(FLAGS.map_file, 'rb') as f:
            char_to_id, id_to_char  = pickle.load(f)

    train_data, test_data, dev_data = prepare_dataset(train_sentences, char_to_id)
    print(train_data[0])
    print(test_data[0])
    print(dev_data[0])
    print(len(train_data),len(dev_data),len(test_data))
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    test_manager = BatchManager(test_data, 100)
    dev_manager = BatchManager(dev_data, 100)

    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)
    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config,logger)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True

    steps_per_epoch = train_manager.len_data

    with tf.Session(config = tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        best = 0
        # sess.graph.finalize()
        for i in range(50):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{},".format(iteration, step%steps_per_epoch, steps_per_epoch))
                    loss = []
            Acc_result = evaluate(sess, model, "dev", dev_manager,  logger)
            logger.info("Acc{}".format(Acc_result))
            logger.info("test")
            # precision, recall, f1_score = model.evaluete_(sess,test_manager)
            # logger.info("P, R, F,{},{},{}".format(precision, recall, f1_score))
            test_result = evaluate(sess, model, "test", test_manager,  logger)
            if test_result > best:
                best = test_result
                save_model(sess, model, FLAGS.ckpt_path, logger)



def input_pro():
    file_name = 'muc_test_result.json'
    with open(file_name, "r", encoding="utf-8") as f:
        line_data = f.readlines()
        return line_data

def write_to_file(result):
    file_path = 'result\\'
    file_name = 'muc_sen_result.json'
    with open(file_path + file_name,'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

def test_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char = pickle.load(f)
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
            logger.info("start testing")
            data = input_pro()
            for line in data:
                data = json.loads(line)
                print(data)
                id = data["id"]
                doc_txt = data["doc"]
                golden_event = data["golden_event"]
                chunk_result = data["chunk_result"]
                result_entity = data["result"]
                for result in result_entity:
                    str_token = result["string"]
                    entities = result["entities"]
                    test_data = input_from_line(str_token, char_to_id)
                    classfy_result = model.evaluate_line(sess, test_data)
                    print(classfy_result)
                    result["mention_classify"] = classfy_result[0]
                write_to_file(data)

                # for sentence_infor in chunk_result:
                #     str_token = sentence_infor["string_token"]
                #     str_chunk = sentence_infor["string_chunk"]
                #     entities = sentence_infor["entities"]
                #     test_data = input_from_line(str_token, char_to_id)
                #     result = model.evaluate_line(sess, test_data)
                #     print(result)
                #     if result[0] == 1:
                #         for entity in entities:
                #             entity_word = entity["word"]
                #             entity["type"] = result

if __name__ == '__main__':
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        test_line()
