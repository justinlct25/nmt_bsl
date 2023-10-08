import scipy.io

# Path to the "features.mat" file
# file_path = './features_5085344787448740525.mat' # 2943s video => 18394 [1*2024]features ~6.25frames/s
# file_path = './features_5085357672350628540.mat' # 3615s video => 22595 [1*1024]features ~6.25frames/s

file_path = './5085344787448740525.mat' # 2943s video => 18394 [1*2024]features ~6.25frames/s


# Load the contents of the ".mat" file
mat_data = scipy.io.loadmat(file_path)

data = mat_data['preds']

print(data.shape)
# print(data)
print(data[139])
print(data[156])
