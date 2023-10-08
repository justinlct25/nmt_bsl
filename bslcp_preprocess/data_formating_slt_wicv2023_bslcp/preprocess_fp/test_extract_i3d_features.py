from operations_i3d import *

# model = model_init()
# features = extract_sign_features("/Volumes/Crucial X8/signlanguge/datasets/bslcp/narrative/BF13n.mov", 50, 74, is_features_per_frame=True)
# features = extract_sign_features("/Volumes/Crucial X8/signlanguge/datasets/bslcp/narrative/BL21n.mov", 50, 52, is_features_per_frame=True)
features = extract_sign_features("/Volumes/Crucial X8/signlanguge/datasets/bslcp/interview/M15i.mov", 50, 52, is_features_per_frame=True, center_square_crop=True)

print(features)
print(features.shape)