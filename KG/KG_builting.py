import pickle
import csv

cls2rel = {"Disease-Position": ["DAP"],
           "Symptom-Position": ["SAP", "SNAP"],
           "Test-Disease": ["TeRD"],
           "Test-Position": ["TeAP", "TeCP"],
           "Test-Symptom": ["TeRS", "TeAS"],
           "Treatment-Disease": ["TrAD", "TrRD"],
           "Treatment-Position": ["TrAP"]}
relations = [[]+v for v in cls2rel.values()]
entity_list = ['Treatment', 'Disease', 'Symptom', 'Test', 'Position']
# 建立实体类的表
# entities = []
# for cls_type in ["Disease-Position", "Symptom-Position", "Test-Position", "Treatment-Position"]:
#     for data in pickle.load(open("./KG/labeled_relations/" + cls_type + ".pkl", "rb")):
#         entities.append(data[0])
# entities = list(set(entities))
# pickle.dump(entities, open("./KG/entites/Position.pkl", "wb"))
# print(entities, len(entities))
for entity_name in entity_list:
    with open("./KG/entities/" + entity_name + ".csv", "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["entity_name"])
        for data in pickle.load(open("./KG/entities/" + entity_name + ".pkl", "rb")):
            f_csv.writerow([data])

# # 建立关系的表
# cls_type = "Treatment-Position"
# rel_name = "TrAP"
# with open("./KG/relations/" + rel_name + ".csv", "w") as f:
#     f_csv = csv.writer(f)
#     f_csv.writerow(["entity_head", "entity_tail"])
#     for data in pickle.load(open("./KG/labeled_relations/" + cls_type + ".pkl", "rb")):
#         if data[1] == rel_name:
#             f_csv.writerow([data[0], data[-1]])
#
# # entities = list(set(entities))
# # pickle.dump(entities, open("./KG/entites/Position.pkl", "wb"))
# # print(entities, len(entities))