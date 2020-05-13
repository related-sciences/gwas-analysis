import numba
import numpy as np
import matplotlib.pyplot as plt

@numba.njit
def rotate(n, m, xi, xj, xv, fv):
    res = np.full((m, n), fv, dtype=np.float32)
    for k in range(len(xi)):
        o = xj[k] - xi[k]
        r, c = m - 1 - o, xi[k] + o
        if r < m and c < n and r >= 0 and c >= 0:
            res[r, c] = xv[k]
    return res

def ld_image(df, n, m, c):
    df = df[(df['i'] < n) & (df['j'] < n)]
    df = df[df['value'].notnull()]
    cols = [df[c].values for c in df]
    img = rotate(n, m, *cols, -.2)
    return img

def ld_plot(img):
    fig, ax = plt.subplots()
    im = ax.imshow(img)
    ax.set_xlabel('Genomic Position')
    fig.set_size_inches(16, 4)