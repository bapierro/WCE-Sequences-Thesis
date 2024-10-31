import os
from wce_cluster import WCECluster
from model_name import Model
path = "../evaluation_data/kvasir_capsule_sequences/selection/selection/ncm_3"




if __name__ ==  "__main__":
    # folders = [f.path for f in os.scandir(path) if f.is_dir()]
    # for folder in folders:
    print("-----------------------")
    # print(f"Section: {folder.split(os.sep)[-1]}")
    WCECluster(path, minCl=[50],backbone=Model.CENDO_FM,sigmas=[5],fps=-4,save_full_fps=True,evaluate=False,recompute=True).apply()


