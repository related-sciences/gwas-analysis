
def png_bytes_to_numpy(png):
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    import numpy as np
    from io import BytesIO
    from PIL import Image
    return np.array(Image.open(BytesIO(png)))


def display_image_grid(images, n_rows, n_cols):
    """Display grid of images provided as numpy as arrays"""
    import numpy as np
    import matplotlib.pyplot as plt
    n = len(images)
    h = np.max([im.shape[0] for im in images])
    w = np.max([im.shape[1] for im in images])
    fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)
    dpi = fig.get_dpi()
    fig.set_size_inches((w*n_cols)/float(dpi), (h*n_rows)/float(dpi))
    for i, ((r, c), ax) in enumerate(np.ndenumerate(axs)):
        ax.axis('off')
        if i == len(images):
            continue
        ax.imshow(images[i])
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)


def display_figure_grid(figures, n_rows, n_cols, fig_width=None, fig_height=None, img_format='png'):
    images = [f.to_image(img_format, width=fig_width, height=fig_height) for f in figures]
    images = list(map(png_bytes_to_numpy, images))
    return display_image_grid(images, n_rows, n_cols)