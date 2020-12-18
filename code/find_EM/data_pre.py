### chunk classify
import json
# data_name = ["train", "test", "dev"]
# for data_set in data_name[0:]:
#     muc_4_structure(data_set)


def write2file(line):
    file_path = "E:\data\MUC-4\muc4_proc\chunk_data\\"
    storage_file_name = "train.json"
    with open(file_path + storage_file_name, "a", encoding='utf-8') as f:
        json.dump(line, f, ensure_ascii=False)
        f.write('\n')

file_path = 'E:\data\MUC-4\muc4_proc\\'
file_name = 'muc_golden_chunk.json'
with open(file_path + file_name, "r", encoding="utf-8") as f:
    line = f.readlines()[0:]

    find_num = 0
    total_num = 0
    for i in range(len(line)):
        data = json.loads(line[i])
        id = data["id"]
        doc_txt = data["doc"]
        golden_events = data["golden_event"]
        chunk_result = data["chunk_result"]
        # print(golden_events)
        golden_list = []
        for golden_event in golden_events:
            for golden in golden_event:
                golden_ = [w.split(" ") for w in golden[1]]
                for golden_entity in golden_:
                    # print([golden_entity, golden[0]])
                    golden_list.append([golden_entity, golden[0]])

        print("**",golden_list)
        total_num += (len(golden_list))
        result_list = []

        entity_list = []
        for sentence_infor in chunk_result:
            string = sentence_infor["string"]
            str_token = sentence_infor["string_token"]
            str_chunk = sentence_infor["string_chunk"]
            entities  = sentence_infor["entities"]
            entities_ = [entity["word"] for entity in entities]
            entity_list += entities_
            for entity in entities_:
                for golden_entity in golden_list:
                    if golden_entity[0][-1] in entity:
                        label_line = [data["id"],entity,str_token,golden_entity[1]]
                        break
                else:
                    label_line = [data["id"],entity,str_token,"Negtive"]
                print(label_line)
                write2file(label_line)
        # print(entity_list)
        chunk_list = [entity for entities in entity_list for entity in entities]
        # print(chunk_list)
        for golden_entity in golden_list:
            if golden_entity[0][-1] in chunk_list:
                find_num += 1

    print(find_num, total_num,find_num/total_num)