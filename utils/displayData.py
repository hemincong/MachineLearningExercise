import numpy as np
import matplotlib.pyplot as plt


def displayData(X, example_width=None):
    """Display 2D data in a nice grid
    [h, display_array] = displayData(X, example_width) displays 2D data
    stored in X in a nice grid.
    :param X:
    :param example_width:
    :return: the figure handle h and the displayed array if requested.
    """
    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(round(np.sqrt(X.shape[1])))

    # Gray Image
    colormap = 'gray'

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            sample_y = pad + j * (example_height + pad)
            sample_x = pad + i * (example_width + pad)
            sample_pixel = X[curr_ex, :].reshape((example_height, example_width), order='F') / max_val
            display_array[sample_y:sample_y + example_height, sample_x:sample_x + example_width] = sample_pixel
            curr_ex += 1
        if curr_ex > m:
            break

    # Display Image
    h = plt.imshow(display_array, cmap=colormap)
    # Do not show axis
    plt.axis('off')

    plt.show()
    return h, display_array
