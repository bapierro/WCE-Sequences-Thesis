
if __name__ == "__main__":
    import os
    from model_name import Model
    from wce_clustering_v2 import WCECluster

    path = "../evaluation_data/AnnotatedVideos_30FPS/e2ec9bac087f4573"
    (WCECluster(
            path,
            minCl=[300],
            sigmas=[4],
            batch_size=32,
            smooth=True,
            fps=30,
            draw_plots=False,
            backbones=[Model.RES_NET_101],
            evaluate=True)
            .apply())
    # folders = [f.path for f in os.scandir(path) if f.is_dir()]
    # for folder in folders:
    #     print("-----------------------")
    #     print(f"Section: {folder.split(os.sep)[-1]}")
        