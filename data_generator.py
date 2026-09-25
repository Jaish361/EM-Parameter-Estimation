import numpy as np


def generate_dataset(
    n_samples=200,
    mean1=2,
    mean2=8,
    std1=0.7,
    std2=0.8
):
    """
    Generate a synthetic dataset
    from two Gaussian distributions.
    """

    n1 = n_samples // 2
    n2 = n_samples - n1

    data1 = np.random.normal(
        mean1,
        std1,
        n1
    )

    data2 = np.random.normal(
        mean2,
        std2,
        n2
    )

    X = np.concatenate(
        [data1, data2]
    )

    np.random.shuffle(X)

    return X