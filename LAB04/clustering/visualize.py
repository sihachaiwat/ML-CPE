import matplotlib.pyplot as plt
import seaborn as sns

def plot_elbow(k_values, inertias, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_clusters(X_2d, labels, save_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette='tab10', s=60, alpha=0.8)
    plt.title('K-Means Animal Clustering')
    plt.xlabel('Weight')
    plt.ylabel('Height')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()