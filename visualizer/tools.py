import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def ebd2rgb(ebds):
    """
    Params:
        embds: embeddings [B, N, C]
    Returns:
        rgbs: [B, N, 3]
    """
    B, N, C = ebds.shape
    pca = PCA(n_components=3)
    scaler = MinMaxScaler()
    
    rgbs = []
    for i, ebd in enumerate(ebds):
        if i == 0:
            rgb = pca.fit_transform(ebd)
            rgb = scaler.fit_transform(rgb)
        else:
            rgb = pca.transform(ebd)
            rgb = scaler.transform(rgb)
        rgbs.append(rgb)

    return np.array(rgbs)

def plot_pcd(pcds, rgbs):
    """
    Params:
        pcds: point clouds [B, N, 3]
        rgbs: [B, N, 3]
    """
    if rgbs is None:
        rgbs = np.ones(pcds.shape)

    images = []
    for i, pcd in enumerate(pcds):
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('on')
        plt.grid(False)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        X = pcd[:, 0]
        Y = pcd[:, 1]
        Z = pcd[:, 2]
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        rgb = rgbs[i]
        rgb = np.clip(rgb, 0., 1.)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], facecolors=np.concatenate([rgb, np.ones((rgb.shape[0], 1))], -1))

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3) / 255
        plt.imsave('./test_vis/{}.png'.format(i), image)
        plt.close(fig)
        images.append(image)
    return np.stack(images, 0)