from tqdm import tqdm
from matplotlib import pyplot as plt
import os

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as skl_sil

from utils import silhouette_coefficient as my_sil



def main():
    # prepare dataset
    iris = load_iris()
    feat_name = iris.feature_names  # feature names
    x, y = iris.data, iris.target

    # preprocess
    os.makedirs('assets', exist_ok=True)

    # plot
    idx = 0
    for i in tqdm(range(len(feat_name))):
        for n in range(i+1, len(feat_name)):
            data = x[:, [i, n]]
            _, ax = plt.subplots(2, 2)
            plt.suptitle(f'Iris Clustering Result', weight='bold')
            plt.tight_layout()
            for k in range(2, 6):
                pred = KMeans(k).fit_predict(data)
                ax[k//2-1, k%2].scatter(data[:, 0], data[:, 1], 
                                        16, pred, cmap='Paired')
                ax[k//2-1, k%2].set_title(f'k={k}, '
                                          f'skl sil={skl_sil(data, pred):.2f}, '
                                          f'my sil={my_sil(data, pred)[0]:.2f}',
                                          size=8,
                                          weight='bold')
                ax[k//2-1, k%2].set_xlabel(feat_name[i])
                ax[k//2-1, k%2].set_ylabel(feat_name[n])
                ax[k//2-1, k%2].set_xticks([])
                ax[k//2-1, k%2].set_yticks([])

            idx += 1
            #plt.show()
            plt.savefig(f'assets/{idx}.png')
            plt.close()



if __name__ == '__main__':
    main()
