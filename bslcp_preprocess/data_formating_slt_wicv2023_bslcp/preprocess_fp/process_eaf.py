
from pympi.Elan import Eaf



def load_eaf_annotations(file_path):
    eaf = Eaf(file_path)
    annotations = []
    for tier_name in eaf.get_tier_names():
        tier = eaf.get_annotation_data_for_tier(tier_name)
        # print(f"Tier: {tier_name}")
        for annotation in tier:
            # print(f"Start: {annotation[0]}, End: {annotation[1]}, Value: {annotation[2]}")
            annotation_dict = {
                "start": annotation[0],
                "end": annotation[1],
                "value": annotation[2]
            }
            annotations.append(annotation_dict)
    return annotations