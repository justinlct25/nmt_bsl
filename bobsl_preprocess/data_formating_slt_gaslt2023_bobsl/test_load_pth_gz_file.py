import gzip
import pickle
import torch

torch.set_printoptions(sci_mode=False) # not printing in e


def load_dataset_file(filename): # to decompress and deserialize the binary file
    with gzip.open(filename, "rb") as f: # the file is being opened in binary mode for reading
        loaded_object = pickle.load(f) # to deserialize the binary data 
        return loaded_object

samples = {}
path = ''
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

print(tmp[0])
# print(len(tmp))
# print(tmp[1]['name'])
# print(tmp[1]['sign'].shape)
# print(tmp[1]['sign'][1][-50:])

# print(tmp[0]['sign'].shape)
# print(tmp[0])
# print(samples['dev/01April_2010_Thursday_heute-6697'])
# print(samples['dev/01April_2010_Thursday_heute-6697']['sign'].shape)
# print(samples['dev/01July_2011_Friday_tagesschau-2763']['sign'].shape)
# print(samples['dev/01July_2011_Friday_tagesschau-2765']['sign'].shape)
# print(samples['dev/01July_2011_Friday_tagesschau-2766']['sign'].shape)
# print(samples['dev/01June_2010_Tuesday_tagesschau-5002']['sign'].shape)
# print(samples['dev/01March_2011_Tuesday_tagesschau-2209']['sign'].shape)
# print(samples['dev/01March_2011_Tuesday_tagesschau-2210']['sign'].shape)
# print(samples['dev/01March_2011_Tuesday_tagesschau-2211']['sign'].shape)

