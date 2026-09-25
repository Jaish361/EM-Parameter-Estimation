import numpy as np


def gaussian_probability(x, mean, variance):
    """
    Calculate Gaussian probability density.
    """

    coefficient = 1 / np.sqrt(2 * np.pi * variance)

    exponent = np.exp(
        -((x - mean) ** 2) / (2 * variance)
    )

    return coefficient * exponent


def calculate_log_likelihood(
    X,
    mean1,
    mean2,
    variance1,
    variance2,
    weight1,
    weight2
):
    """
    Calculate the log-likelihood of the dataset.
    """

    probability = (
        weight1 * gaussian_probability(
            X, mean1, variance1
        )
        +
        weight2 * gaussian_probability(
            X, mean2, variance2
        )
    )

    # Prevent log(0)
    probability = np.maximum(
        probability,
        1e-300
    )

    return np.sum(np.log(probability))


def e_step(
    X,
    mean1,
    mean2,
    variance1,
    variance2,
    weight1,
    weight2
):
    """
    Expectation Step.

    Calculates responsibilities for
    every data point.
    """

    responsibilities = []

    for x in X:

        # Probability under each component
        p1 = gaussian_probability(
            x,
            mean1,
            variance1
        )

        p2 = gaussian_probability(
            x,
            mean2,
            variance2
        )

        # Include mixing weights
        weighted_p1 = weight1 * p1
        weighted_p2 = weight2 * p2

        total = weighted_p1 + weighted_p2

        # Responsibilities
        r1 = weighted_p1 / total
        r2 = weighted_p2 / total

        responsibilities.append([r1, r2])

    return np.array(responsibilities)


def m_step(X, responsibilities):
    """
    Maximization Step.

    Updates:
    - Mean
    - Variance
    - Mixing weight
    """

    # Effective number of points
    N1 = np.sum(
        responsibilities[:, 0]
    )

    N2 = np.sum(
        responsibilities[:, 1]
    )

    # -------------------------
    # Update means
    # -------------------------

    new_mean1 = (
        np.sum(
            responsibilities[:, 0] * X
        ) / N1
    )

    new_mean2 = (
        np.sum(
            responsibilities[:, 1] * X
        ) / N2
    )

    # -------------------------
    # Update variances
    # -------------------------

    new_variance1 = (
        np.sum(
            responsibilities[:, 0]
            * (X - new_mean1) ** 2
        ) / N1
    )

    new_variance2 = (
        np.sum(
            responsibilities[:, 1]
            * (X - new_mean2) ** 2
        ) / N2
    )

    # -------------------------
    # Update weights
    # -------------------------

    N = len(X)

    new_weight1 = N1 / N
    new_weight2 = N2 / N

    return (
        new_mean1,
        new_mean2,
        new_variance1,
        new_variance2,
        new_weight1,
        new_weight2
    )


def run_em(
    X,
    max_iterations=20,
    tolerance=0.0001
):
    """
    Complete Expectation-Maximization algorithm.
    """

    # -------------------------
    # Initial parameters
    # -------------------------

    mean1 = 3.0
    mean2 = 6.0

    variance1 = 1.0
    variance2 = 1.0

    weight1 = 0.5
    weight2 = 0.5

    history = []

    # -------------------------
    # EM iterations
    # -------------------------

    for iteration in range(max_iterations):

        old_mean1 = mean1
        old_mean2 = mean2

        old_variance1 = variance1
        old_variance2 = variance2

        # =====================
        # E-STEP
        # =====================

        responsibilities = e_step(
            X,
            mean1,
            mean2,
            variance1,
            variance2,
            weight1,
            weight2
        )

        # =====================
        # M-STEP
        # =====================

        (
            mean1,
            mean2,
            variance1,
            variance2,
            weight1,
            weight2
        ) = m_step(
            X,
            responsibilities
        )

        # =====================
        # LOG-LIKELIHOOD
        # =====================

        log_likelihood = calculate_log_likelihood(
            X,
            mean1,
            mean2,
            variance1,
            variance2,
            weight1,
            weight2
        )

        # Store iteration
        history.append({
            "iteration": iteration + 1,
            "mean1": mean1,
            "mean2": mean2,
            "variance1": variance1,
            "variance2": variance2,
            "weight1": weight1,
            "weight2": weight2,
            "log_likelihood": log_likelihood
        })

        # =====================
        # CONVERGENCE
        # =====================

        change = max(
            abs(mean1 - old_mean1),
            abs(mean2 - old_mean2),
            abs(variance1 - old_variance1),
            abs(variance2 - old_variance2)
        )

        if change < tolerance:
            break

    return {
        "mean1": mean1,
        "mean2": mean2,
        "variance1": variance1,
        "variance2": variance2,
        "weight1": weight1,
        "weight2": weight2,
        "responsibilities": responsibilities,
        "history": history,
        "iterations": iteration + 1
    }