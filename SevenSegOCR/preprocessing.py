import pywt
import matplotlib.pyplot as plt


def preprocess(img):
    LL, (LH, HL, HH) = pywt.dwt2(img, 'bior1.3')
    fig = plt.figure(figsize=(12, 3))

    titles = ['Original', 'Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    for i, a in enumerate([img, LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 5, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
