from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from mne.decoding import Vectorizer


def tsne(X, label, n_components):

    # Vectorization
    vect = Vectorizer()
    X = vect.fit_transform(X)

    # TSNE
    perplexities = [6, 11, 18, 25]

    for i, perplexity in enumerate(tqdm(perplexities)):
        model = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
        features_tsne = model.fit_transform(X)

        if n_components == 2:
            plt.figure(figsize=(10, 6))

            x = features_tsne[:, 0]
            y = features_tsne[:, 1]

            plt.scatter(x, y, c=label, cmap='Paired')
            plt.title(f"Perplexity = {perplexity}")

            plt.xticks([])
            plt.yticks([])

        elif n_components == 3:
            plt.figure(figsize=(10, 6))
            ax = plt.axes(projection='3d')

            x = features_tsne[:, 0]
            y = features_tsne[:, 1]
            z = features_tsne[:, 2]

            ax.scatter(x, y, z, c=label, cmap='Paired')
            ax.set_title(f"Perplexity = {perplexity}")

    plt.show()


if __name__ == '__main__':
    components = 3
    x, y = 1, 2
    tsne(x, y, components)
