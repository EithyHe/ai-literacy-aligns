import numpy as np
import umap
import matplotlib.pyplot as plt

def run_umap_from_scores(
    pca_scores_npy: str,
    labels_csv: str,
    out_png: str,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42
):
    X = np.load(pca_scores_npy)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    Y = reducer.fit_transform(X)
    plt.figure(figsize=(8,6), dpi=140)
    plt.scatter(Y[:,0], Y[:,1], s=6, alpha=0.75)
    plt.title("UMAP on PCA scores")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
