import gzip
import pickle
import torch

torch.set_printoptions(sci_mode=False) # not printing in e


def load_dataset_file(filename): # to decompress and deserialize the binary file
    with gzip.open(filename, "rb") as f: # the file is being opened in binary mode for reading
        loaded_object = pickle.load(f) # to deserialize the binary data 
        return loaded_object

samples = {}
## total: 8257 ratio: 0.86:0.063:0.078
# path = "../features_phoenix2014t/phoenix14t.pami0.train" # 7096
# path = "../features_phoenix2014t/phoenix14t.pami0.dev" # 519
# path = "../features_phoenix2014t/phoenix14t.pami0.test" # 642
# path = "./dataset1111.pth.gz"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/dataset_bobsl_29episodes_0.7_googlenet/phoenix14t.pami0.dev"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_44episodes_0.7_googlenet(filtered_0gloss_1minglossmatch)/dataset.pth.gz.dev"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_56episodes_0.7_i3d(filtered_0gloss)/phoenix14t.pami0.dev"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_55episodes_0.7_i3d_filtered_0gloss_1minmatch/dataset.pth.gz.dev"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_55episodes_0.3_i3d_stemC_filtered_0glossS_<1glossmatchedS_<0.1glosstxtratioS_0appearG(11216)/phoenix14t.pami0.test"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/features_phoenix2014t/phoenix14t.pami0.train"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_55episodes_0.3_i3d_stemC_filtered_<0glossmatchedS_<0glosstxtratioS_0appearG_<0words_>(inf)words_sentence(31479)_new/dataset.pth.gz.train"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_29episodes_0.7_googlenet(unfiltered)/phoenix14t.pami0.train"
# path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_55episodes_0.7_googlenet_stemC_filtered_0glossS_<1glossmatchedS_0appearG(14421)/dataset.pth.gz.train"
path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_55episodes_0.5_i3d_stemC_filtered_0glossS_<1glossmatchedS_<0glosstxtratioS_0appearG_>30words_sentence(14633)_new/dataset.pth.gz.train"
tmp = load_dataset_file(path)
seq_id = ""
for s in tmp:
    seq_id = s["name"]
    if seq_id in samples:
        assert samples[seq_id]["name"] == s["name"]
        assert samples[seq_id]["signer"] == s["signer"]
        assert samples[seq_id]["gloss"] == s["gloss"]
        assert samples[seq_id]["text"] == s["text"]
        samples[seq_id]["sign"] = torch.cat(
            [samples[seq_id]["sign"], s["sign"]], axis=1
        )
    else: # all go to this
        samples[seq_id] = {
            "name": s["name"],
            "signer": s["signer"],
            "gloss": s["gloss"],
            "text": s["text"],
            "sign": s["sign"],
        }

# print(seq_id+":")
# print(samples[seq_id])

# print(tmp[1]['name'])
# print(tmp[1])
# print(tmp[1]['sign'].shape)
# print(tmp[1]['sign'])
# print(tmp[1]['sign'][1][-50:])

# print(tmp[1]['sign'][0].shape)
# print(tmp[1]['sign'][0].numpy().tolist())

# print(tmp[1])
# print(len(tmp))
# print(type(tmp))
# print(type(tmp[0]['sign']))
# print(samples)
# print(tmp[0]['sign'].shape[0])

# for i, s in enumerate(tmp):
#     if len(s["gloss"].split(" "))/len(s["text"].split(" ")) > 0.5 and len(s["text"].split(" ")) > 5:
#         print(s)
#         print(s["sign"].shape)

# count avg gloss density
gloss_density_sum = 0
for i, s in enumerate(tmp):
    gloss_density_sum += len(s["gloss"].split(" "))/len(s["text"].split(" "))
avg_gloss_density = gloss_density_sum/len(tmp)
print(avg_gloss_density)
