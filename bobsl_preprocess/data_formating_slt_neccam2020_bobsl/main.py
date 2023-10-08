from googlenet_pytorch import GoogLeNet



model = GoogLeNet.from_pretrained("googlenet")
model.eval()