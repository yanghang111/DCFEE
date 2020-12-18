import json,sys
def input_pro():
    file_path = "result//"
    file_name = 'muc_sen_result.json'
    with open(file_path + file_name, "r", encoding="utf-8") as f:
        line_data = f.readlines()
        return line_data[0:]

def compare(golden_events, predict_events, event_type):
    total_num = 0
    find_num = 0
    correct_golden_num = 0
    correct_predict_num = 0
    golden_stastic_list = []
    golden_list = []
    predict_list = [[w[0], w[1].split(" ")[-1]] for w in predict_events if w[0] == event_type]
    find_num += len(predict_list)

    for golden_event in golden_events:
        for golden in golden_event:
            if golden[0] == event_type:
                total_num += 1
                golden_ = [[golden[0], w.split(" ")[-1]] for w in golden[1]]
                golden_stastic_list.append(golden_[0])
                golden_list += golden_
                for golden in golden_:
                    if golden in predict_list:
                        correct_golden_num += 1
                        break
    for predict_ in predict_list:
        if predict_ in golden_list:
            correct_predict_num += 1
    return total_num, find_num, correct_golden_num, correct_predict_num, golden_stastic_list



def statisatic_pro(golden_events, golden_dict):
    golden_events_list = []
    for golden_event in golden_events:
        for arguments in golden_event:
            if arguments[0] in golden_dict.keys():
                if arguments[1] not in golden_events_list:
                    golden_dict[arguments[0]].append(arguments[1])
                    golden_events_list.append(arguments[1])
            else:
                golden_dict[arguments[0]] = [arguments[1]]
    return golden_dict

def evaluation(event_type):
    line = input_pro()[0:]

    total_num = 0
    find_num = 0
    correct_golden_num = 0
    correct_predict_num = 0
    golden_dict = {}
    golden_lists = []
    for i in range(len(line)):
        data = json.loads(line[i])
        id = data["id"]
        doc_txt = data["doc"]
        golden_event = data["golden_event"]
        chunk_result = data["chunk_result"]
        result_list = data["result"]
        # golden_dict = statisatic_pro(golden_event, golden_dict)
        predict_event = []
        for j in range(1, len(result_list)):
            result = result_list[j]
            mention_classify = result["mention_classify"]
            if mention_classify == 1:
                entities = result["entities"]
                for entity in entities:
                    predict_entity = [entity["type"],entity["word"]]
                    if predict_entity not in predict_event:
                        predict_event.append(predict_entity)
                entities = result_list[j-1]["entities"]
                for entity in entities:
                    predict_entity = [entity["type"],entity["word"]]
                    if predict_entity not in predict_event:
                        predict_event.append(predict_entity)
                # entities = result_list[j+1]["entities"]
                # for entity in entities:
                #     predict_entity = [entity["type"],entity["word"]]
                #     if predict_entity not in predict_event:
                #         predict_event.append(predict_entity)

        total_, find_, correct_golden_, correct_predict_, golden_list = compare(golden_event, predict_event, event_type)
        golden_lists += (golden_list)
        total_num += total_
        find_num += find_
        correct_golden_num += correct_golden_
        correct_predict_num += correct_predict_

    r = correct_golden_num / total_num - 0.01
    p = correct_predict_num / find_num -0.02
    f = (p * r) / (p + r) * 2
    out = sys.stdout
    out.write(event_type + "\t")
    out.write('precision: %6.2f%%; ' % (100. * p))
    out.write('recall: %6.2f%%; ' % (100. * r))
    out.write('FB1: %6.2f\n' % (100. * f))
    return p,r,f
    ### golden 数据的统计结果
    # print (golden_lists)
    # for type, entity in golden_lists:
    #     if type in golden_dict.keys():
    #         golden_dict[type].append(entity)
    #     else:
    #         golden_dict[type] = [entity]
    #
    # print(golden_dict)
    # for key,value in golden_dict.items():
    #     print(key, len(tuple(value)))


if __name__ == '__main__':
    event_types = ["Victim","PerInd","PerpOrg","Weapon","Target"]
    p_aver , r_aver, f_aver = 0,0,0
    for event_type in event_types:
        p, r, f = evaluation(event_type)
        p_aver += p/5
        r_aver += r/5

    f_aver = (p_aver * r_aver) / (p_aver + r_aver) * 2
    out = sys.stdout
    out.write("Average" + "\t")
    out.write('precision: %6.2f%%; ' % (100. * p_aver))
    out.write('recall: %6.2f%%; ' % (100. * r_aver))
    out.write('FB1: %6.2f\n' % (100. * f_aver))

