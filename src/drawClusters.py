import os
from wce_clustering_v2 import WCECluster
from model_name import Model
path = "../evaluation_data/kvasir_capsule_sequences/selection/selection/ncm_6"



if __name__ ==  "__main__":
    # folders = [f.path for f in os.scandir(path) if f.is_dir()]
    # for folder in folders:
    print("-----------------------")
    # print(f"Section: {folder.split(os.sep)[-1]}")
    WCECluster(path,save_representatives=False,backbones=[Model.CENDO_FM],sigmas=[3],fps=10,evaluate=True, external_validation=True,recompute=False,draw_plot=True).apply()


