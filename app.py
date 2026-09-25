import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_generator import generate_dataset
from em_algorithm import run_em, gaussian_probability


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EM Parameter Estimation",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 EM Parameter Estimation Demonstration")

st.markdown(
    """
    ### Expectation-Maximization Algorithm

    This application demonstrates how the **Expectation-Maximization
    (EM) algorithm** estimates the parameters of two hidden Gaussian
    distributions from observed data.

    **Main focus:** Understanding the **E-Step and M-Step**, especially
    how the M-Step updates the estimated parameters.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Experiment Settings")

n_samples = st.sidebar.slider(
    "Number of Data Points",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

max_iterations = st.sidebar.slider(
    "Maximum EM Iterations",
    min_value=1,
    max_value=50,
    value=20
)

tolerance = st.sidebar.number_input(
    "Convergence Tolerance",
    min_value=0.00001,
    max_value=0.1,
    value=0.0001,
    format="%.5f"
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### Dataset

    The application generates data from two hidden Gaussian
    distributions:

    **Component 1**
    - Mean ≈ 2
    - Standard deviation ≈ 0.7

    **Component 2**
    - Mean ≈ 8
    - Standard deviation ≈ 0.8

    The EM algorithm does not receive these values.
    It estimates them from the observed data.
    """
)


# ============================================================
# RUN EM
# ============================================================

if st.sidebar.button(
    "🚀 Run EM Algorithm",
    use_container_width=True
):

    # Generate dataset
    X = generate_dataset(
        n_samples=n_samples
    )

    # Run EM
    result = run_em(
        X,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    # Store results
    st.session_state["X"] = X
    st.session_state["result"] = result


# ============================================================
# INITIAL SCREEN
# ============================================================

if "result" not in st.session_state:

    st.info(
        "👈 Configure the experiment from the sidebar "
        "and click **Run EM Algorithm** to begin."
    )

    st.subheader("🔄 How the Algorithm Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### 1️⃣ E-Step

            Calculate the probability that each data point
            belongs to each Gaussian component.

            These probabilities are called
            **responsibilities**.
            """
        )

    with col2:
        st.markdown(
            """
            ### 2️⃣ M-Step

            Use the responsibilities as weights to update:

            - Mean
            - Variance
            - Mixing weight
            """
        )

    with col3:
        st.markdown(
            """
            ### 3️⃣ Repeat

            E-Step and M-Step are repeated until the
            parameters become stable.

            This is called **convergence**.
            """
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "result" in st.session_state:

    X = st.session_state["X"]
    result = st.session_state["result"]

    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        f"✅ EM algorithm completed in "
        f"{result['iterations']} iterations."
    )


    # ========================================================
    # SECTION 1 — FINAL PARAMETERS
    # ========================================================

    st.header("1️⃣ Estimated Parameters")

    parameter_table = pd.DataFrame({
        "Component": [
            "Component 1",
            "Component 2"
        ],

        "Mean": [
            result["mean1"],
            result["mean2"]
        ],

        "Variance": [
            result["variance1"],
            result["variance2"]
        ],

        "Weight": [
            result["weight1"],
            result["weight2"]
        ]
    })

    st.dataframe(
        parameter_table.style.format({
            "Mean": "{:.4f}",
            "Variance": "{:.4f}",
            "Weight": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean 1",
            f"{result['mean1']:.3f}"
        )

    with col2:
        st.metric(
            "Mean 2",
            f"{result['mean2']:.3f}"
        )

    with col3:
        st.metric(
            "Iterations",
            result["iterations"]
        )

    with col4:
        st.metric(
            "Data Points",
            len(X)
        )


    st.divider()


    # ========================================================
    # SECTION 2 — OBSERVED DATA
    # ========================================================

    st.header("2️⃣ Observed Dataset")

    st.markdown(
        """
        This histogram shows the observed data given to the EM
        algorithm. The algorithm does not know which data point
        came from which hidden distribution.
        """
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        X,
        bins=30,
        density=True,
        alpha=0.6
    )

    ax.set_xlabel("Data Value")
    ax.set_ylabel("Density")
    ax.set_title("Observed Data Distribution")

    ax.grid(True, alpha=0.2)

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # SECTION 3 — GAUSSIAN COMPONENTS
    # ========================================================

    st.header("3️⃣ Estimated Gaussian Components")

    x_values = np.linspace(
        X.min() - 1,
        X.max() + 1,
        500
    )

    # Component 1
    curve1 = (
        result["weight1"]
        * gaussian_probability(
            x_values,
            result["mean1"],
            result["variance1"]
        )
    )

    # Component 2
    curve2 = (
        result["weight2"]
        * gaussian_probability(
            x_values,
            result["mean2"],
            result["variance2"]
        )
    )

    # Combined mixture
    mixture = curve1 + curve2

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        X,
        bins=30,
        density=True,
        alpha=0.3,
        label="Observed Data"
    )

    ax.plot(
        x_values,
        curve1,
        linewidth=2,
        label="Gaussian Component 1"
    )

    ax.plot(
        x_values,
        curve2,
        linewidth=2,
        label="Gaussian Component 2"
    )

    ax.plot(
        x_values,
        mixture,
        linewidth=3,
        linestyle="--",
        label="Combined Gaussian Mixture"
    )

    ax.set_xlabel("Data Value")
    ax.set_ylabel("Density")
    ax.set_title("EM Estimated Gaussian Mixture")

    ax.legend()
    ax.grid(True, alpha=0.2)

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        """
        **Interpretation:** The histogram represents the observed data.
        The two solid curves represent the Gaussian components estimated
        by EM. The dashed curve represents their combined mixture.
        """
    )


    # ========================================================
    # SECTION 4 — E-STEP
    # ========================================================

    st.header("4️⃣ E-Step — Calculate Responsibilities")

    st.markdown(
        """
        In the **E-Step**, EM calculates how strongly each data point
        belongs to each Gaussian component.

        These values are called **responsibilities**.
        """
    )

    responsibilities = result["responsibilities"]

    responsibility_df = pd.DataFrame({
        "Data Point": X[:10],
        "Component 1 Responsibility": responsibilities[:10, 0],
        "Component 2 Responsibility": responsibilities[:10, 1]
    })

    st.dataframe(
        responsibility_df.style.format({
            "Data Point": "{:.4f}",
            "Component 1 Responsibility": "{:.4f}",
            "Component 2 Responsibility": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing the first 10 data points."
    )


    # ========================================================
    # SECTION 5 — M-STEP
    # ========================================================

    st.header("5️⃣ M-Step — Parameter Estimation")

    st.markdown(
        """
        ### What happens in the M-Step?

        The responsibilities calculated during the E-Step are used
        as **weights**.

        These weighted observations are then used to estimate new:

        - **Mean (μ)**
        - **Variance (σ²)**
        - **Mixing Weight (π)**
        """
    )

    st.code(
        """
New Mean =
Σ(responsibility × data)
─────────────────────────
Σ(responsibility)


New Variance =
Σ[responsibility × (data - new_mean)²]
─────────────────────────────────────
Σ(responsibility)


New Weight =
Σ(responsibility)
────────────────
Number of data points
        """,
        language="text"
    )

    st.success(
        """
        **Key idea:** The responsibility acts as a weight.
        Data points with higher responsibility have greater influence
        when estimating the new parameters.
        """
    )


    # ========================================================
    # SECTION 6 — PARAMETER CONVERGENCE
    # ========================================================

    st.header("6️⃣ Parameter Convergence")

    history_df = pd.DataFrame(
        result["history"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        history_df["iteration"],
        history_df["mean1"],
        marker="o",
        label="Estimated Mean 1"
    )

    ax.plot(
        history_df["iteration"],
        history_df["mean2"],
        marker="o",
        label="Estimated Mean 2"
    )

    ax.axhline(
        2,
        linestyle="--",
        label="Underlying Mean ≈ 2"
    )

    ax.axhline(
        8,
        linestyle="--",
        label="Underlying Mean ≈ 8"
    )

    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Mean")
    ax.set_title("Mean Parameter Convergence")

    ax.legend()
    ax.grid(True, alpha=0.2)

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # SECTION 7 — LOG-LIKELIHOOD
    # ========================================================

    st.header("7️⃣ Log-Likelihood")

    st.markdown(
        """
        Log-likelihood measures how well the current estimated
        parameters explain the observed data.

        During EM optimization, the likelihood generally improves
        and eventually stabilizes as the algorithm converges.
        """
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        history_df["iteration"],
        history_df["log_likelihood"],
        marker="o"
    )

    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Log-Likelihood Across EM Iterations")

    ax.grid(True, alpha=0.2)

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # SECTION 8 — ITERATION HISTORY
    # ========================================================

    st.header("8️⃣ EM Iteration History")

    display_history = history_df.copy()

    st.dataframe(
        display_history.style.format({
            "mean1": "{:.4f}",
            "mean2": "{:.4f}",
            "variance1": "{:.4f}",
            "variance2": "{:.4f}",
            "weight1": "{:.4f}",
            "weight2": "{:.4f}",
            "log_likelihood": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SECTION 9 — FINAL EXPLANATION
    # ========================================================

    st.header("9️⃣ What Did EM Learn?")

    st.markdown(
        f"""
        ### Final Result

        The EM algorithm estimated two hidden Gaussian components.

        **Component 1**

        - Estimated mean: **{result['mean1']:.4f}**
        - Estimated variance: **{result['variance1']:.4f}**
        - Estimated weight: **{result['weight1']:.4f}**

        **Component 2**

        - Estimated mean: **{result['mean2']:.4f}**
        - Estimated variance: **{result['variance2']:.4f}**
        - Estimated weight: **{result['weight2']:.4f}**

        The algorithm repeatedly performed:

        **E-Step → M-Step → E-Step → M-Step**

        until the parameter changes became sufficiently small.
        """
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "EM Parameter Estimation Demonstration | "
        "Machine Learning Project"
    )