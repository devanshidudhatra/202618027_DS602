# Streamlit Dashboard for Lab-4
# Medical Insurance Statistical Modeling

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Insurance Statistical Lab",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.05


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")


# ============================================================
# REGRESSION MODEL
# ============================================================

@st.cache_resource
def fit_model(df):

    formula = (
        "charges ~ age + bmi + children "
        "+ C(sex) + C(smoker) + C(region)"
    )

    model = smf.ols(
        formula=formula,
        data=df
    ).fit()

    return model


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def descriptive(df):

    numeric = df.select_dtypes(include=np.number)

    result = pd.DataFrame({
        "Mean": numeric.mean(),
        "Median": numeric.median(),
        "Std Dev": numeric.std(),
        "IQR": numeric.quantile(0.75) - numeric.quantile(0.25),
        "Skewness": numeric.skew(),
        "Kurtosis": numeric.kurtosis()
    })

    return result.round(3)


# ============================================================
# HYPOTHESIS TEST 1
# ============================================================

def hypothesis_1(df, group_col, metric):

    levels = df[group_col].dropna().unique().tolist()

    if len(levels) != 2:
        raise ValueError(
            "Hypothesis Test 1 requires exactly two groups."
        )

    group_a = df.loc[
        df[group_col] == levels[0],
        metric
    ].dropna()

    group_b = df.loc[
        df[group_col] == levels[1],
        metric
    ].dropna()

    # Shapiro-Wilk normality test
    shapiro_a = stats.shapiro(group_a)
    shapiro_b = stats.shapiro(group_b)

    # Levene's test for equal variances
    levene_test = stats.levene(
        group_a,
        group_b,
        center="median"
    )

    # Check normality
    normal = (
        shapiro_a.pvalue > ALPHA
        and
        shapiro_b.pvalue > ALPHA
    )

    # Select appropriate test
    if normal:

        equal_variance = levene_test.pvalue > ALPHA

        result = stats.ttest_ind(
            group_a,
            group_b,
            equal_var=equal_variance
        )

        if equal_variance:
            test_name = "Independent two-sample t-test"
        else:
            test_name = "Welch two-sample t-test"

        statistic = result.statistic
        p_value = result.pvalue

    else:

        result = stats.mannwhitneyu(
            group_a,
            group_b,
            alternative="two-sided"
        )

        test_name = "Mann-Whitney U test"

        statistic = result.statistic
        p_value = result.pvalue

    return {
        "group_a": levels[0],
        "group_b": levels[1],
        "n_a": len(group_a),
        "n_b": len(group_b),
        "mean_a": group_a.mean(),
        "mean_b": group_b.mean(),
        "shapiro_a_p": shapiro_a.pvalue,
        "shapiro_b_p": shapiro_b.pvalue,
        "levene_p": levene_test.pvalue,
        "normal": normal,
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value
    }


# ============================================================
# CHI-SQUARE TEST
# ============================================================

def chi_square(df, cat1, cat2):

    table = pd.crosstab(
        df[cat1],
        df[cat2]
    )

    chi2, p_value, dof, expected = stats.chi2_contingency(
        table
    )

    return table, chi2, p_value, dof


# ============================================================
# ANOVA
# ============================================================

def anova(df, category, metric):

    groups = []

    for _, group in df.groupby(category):

        values = group[metric].dropna().values

        if len(values) > 0:
            groups.append(values)

    if len(groups) < 2:
        return None, None

    f_statistic, p_value = stats.f_oneway(*groups)

    return f_statistic, p_value


# ============================================================
# Q-Q PLOT
# ============================================================

def qq_plot(residuals):

    theoretical, ordered = stats.probplot(
        residuals,
        dist="norm",
        fit=False
    )

    # Calculate reference line
    slope, intercept = np.polyfit(
        theoretical,
        ordered,
        1
    )

    x = np.array(theoretical)

    fig = go.Figure()

    # Q-Q points
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=ordered,
            mode="markers",
            name="Residuals"
        )
    )

    # Reference line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=intercept + slope * x,
            mode="lines",
            name="Reference line"
        )
    )

    fig.update_layout(
        title="Q-Q Plot of Residuals",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Ordered Residuals"
    )

    return fig


# ============================================================
# PREDICTION INTERVAL
# ============================================================

def prediction_interval(model, row_df):

    prediction = model.get_prediction(
        row_df
    ).summary_frame(alpha=ALPHA)

    return prediction


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏥 Medical Insurance — Applied Statistical Modeling"
)

st.caption(
    "DS602 Lab-4 | Part 1: EDA & Hypothesis Testing | "
    "Part 2: OLS & Diagnostics"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Data Exploration",
        "🧪 Hypothesis Testing Lab",
        "📈 Live Prediction & Diagnostics"
    ]
)


# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================

with tab1:

    st.header("Data Exploration")

    # --------------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------------

    st.sidebar.header("Filters")

    age_min = int(df["age"].min())
    age_max = int(df["age"].max())

    age_range = st.sidebar.slider(
        "Age range",
        age_min,
        age_max,
        (age_min, age_max)
    )

    charge_min = float(df["charges"].min())
    charge_max = float(df["charges"].max())

    charge_range = st.sidebar.slider(
        "Charges range",
        charge_min,
        charge_max,
        (charge_min, charge_max)
    )

    smoker_options = sorted(
        df["smoker"].unique()
    )

    smoker_filter = st.sidebar.multiselect(
        "Smoker",
        smoker_options,
        default=smoker_options
    )

    region_options = sorted(
        df["region"].unique()
    )

    region_filter = st.sidebar.multiselect(
        "Region",
        region_options,
        default=region_options
    )

    sex_options = sorted(
        df["sex"].unique()
    )

    sex_filter = st.sidebar.multiselect(
        "Sex",
        sex_options,
        default=sex_options
    )


    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    filtered = df[
        df["age"].between(
            age_range[0],
            age_range[1]
        )
        &
        df["charges"].between(
            charge_range[0],
            charge_range[1]
        )
        &
        df["smoker"].isin(smoker_filter)
        &
        df["region"].isin(region_filter)
        &
        df["sex"].isin(sex_filter)
    ].copy()


    # --------------------------------------------------------
    # CHECK EMPTY DATA
    # --------------------------------------------------------

    if len(filtered) == 0:

        st.warning(
            "No records match the selected filters. "
            "Please change the filters."
        )

    else:

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows after filters",
            len(filtered)
        )

        c2.metric(
            "Mean charges",
            f"${filtered['charges'].mean():,.0f}"
        )

        c3.metric(
            "Median charges",
            f"${filtered['charges'].median():,.0f}"
        )


        # ----------------------------------------------------
        # DESCRIPTIVE STATISTICS
        # ----------------------------------------------------

        st.subheader(
            "Descriptive Statistics"
        )

        st.dataframe(
            descriptive(filtered),
            use_container_width=True
        )


        # ----------------------------------------------------
        # DISTRIBUTION + SCATTER
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        # Distribution
        with col1:

            metric = st.selectbox(
                "Distribution metric",
                [
                    "age",
                    "bmi",
                    "children",
                    "charges"
                ]
            )

            fig = px.histogram(
                filtered,
                x=metric,
                marginal="box",
                nbins=30,
                title=f"Distribution of {metric}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # Scatter plot
        with col2:

            x_variable = st.selectbox(
                "Scatter X",
                [
                    "age",
                    "bmi",
                    "children"
                ],
                index=0
            )

            fig = px.scatter(
                filtered,
                x=x_variable,
                y="charges",
                color="smoker",
                trendline="ols",
                title=f"{x_variable} vs Charges"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # CORRELATION MATRIX
        # ----------------------------------------------------

        st.subheader(
            "Correlation Matrix"
        )

        correlation = filtered.select_dtypes(
            include=np.number
        ).corr()

        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            title="Numerical Correlations"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 2 — HYPOTHESIS TESTING
# ============================================================

with tab2:

    st.header(
        "Hypothesis Testing Lab"
    )

    st.write(
        f"Significance level: **α = {ALPHA}**"
    )


    # ========================================================
    # HYPOTHESIS TEST 1
    # ========================================================

    st.subheader(
        "Hypothesis Test 1 — Compare Two Groups"
    )

    categorical_cols = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()


    group_col = st.selectbox(
        "Categorical grouping variable",
        categorical_cols,
        index=(
            categorical_cols.index("smoker")
            if "smoker" in categorical_cols
            else 0
        )
    )


    metric_col = st.selectbox(
        "Numerical metric",
        numeric_cols,
        index=(
            numeric_cols.index("charges")
            if "charges" in numeric_cols
            else 0
        )
    )


    # Check whether exactly two groups exist
    if df[group_col].nunique() == 2:

        h1 = hypothesis_1(
            df,
            group_col,
            metric_col
        )


        st.write(
            f"**H0:** {metric_col} has no difference "
            f"between the two groups."
        )

        st.write(
            f"**H1:** {metric_col} differs "
            f"between the two groups."
        )


        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Test",
            h1["test"]
        )

        m2.metric(
            "Test Statistic",
            f"{h1['statistic']:.4f}"
        )

        m3.metric(
            "p-value",
            f"{h1['p_value']:.4g}"
        )


        st.info(
            f"Shapiro p-values: "
            f"{h1['group_a']} = {h1['shapiro_a_p']:.4g}, "
            f"{h1['group_b']} = {h1['shapiro_b_p']:.4g} "
            f"| Levene p-value = {h1['levene_p']:.4g}"
        )


        if h1["p_value"] < ALPHA:

            st.success(
                "Reject H0: there is statistically "
                "significant evidence of a difference."
            )

        else:

            st.warning(
                "Fail to reject H0: insufficient "
                "evidence of a difference."
            )


        st.write(
            f"**{h1['group_a']} mean:** "
            f"{h1['mean_a']:.2f}"
        )

        st.write(
            f"**{h1['group_b']} mean:** "
            f"{h1['mean_b']:.2f}"
        )


    else:

        st.warning(
            "Choose a categorical variable with "
            "exactly two levels for Hypothesis Test 1."
        )


    # ========================================================
    # HYPOTHESIS TEST 2
    # ========================================================

    st.divider()

    st.subheader(
        "Hypothesis Test 2 — BOTH Required Tests"
    )


    # ========================================================
    # 2A. CHI-SQUARE TEST
    # ========================================================

    st.markdown(
        "### A. Chi-Square Test of Independence"
    )

    cat1 = st.selectbox(
        "Categorical variable 1",
        categorical_cols,
        index=(
            categorical_cols.index("smoker")
            if "smoker" in categorical_cols
            else 0
        ),
        key="chi1"
    )


    cat2 = st.selectbox(
        "Categorical variable 2",
        categorical_cols,
        index=(
            categorical_cols.index("region")
            if "region" in categorical_cols
            else 0
        ),
        key="chi2"
    )


    if cat1 == cat2:

        st.warning(
            "Please select two different categorical variables."
        )

    else:

        table, chi_stat, chi_p, chi_dof = chi_square(
            df,
            cat1,
            cat2
        )


        st.write(
            "**H0:** The two categorical variables are independent."
        )

        st.write(
            "**H1:** The two categorical variables are associated."
        )


        st.write(
            "Contingency Table:"
        )

        st.dataframe(
            table,
            use_container_width=True
        )


        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Chi-square",
            f"{chi_stat:.4f}"
        )

        c2.metric(
            "Degrees of Freedom",
            chi_dof
        )

        c3.metric(
            "p-value",
            f"{chi_p:.4g}"
        )


        # IMPORTANT:
        # Normal if/else instead of one-line conditional.
        # This prevents DeltaGenerator from appearing.

        if chi_p < ALPHA:

            st.success(
                "Reject H0: variables are associated."
            )

        else:

            st.warning(
                "Fail to reject H0: insufficient evidence "
                "of an association."
            )


    # ========================================================
    # 2B. ONE-WAY ANOVA
    # ========================================================

    st.markdown(
        "### B. One-Way ANOVA"
    )


    anova_group = st.selectbox(
        "Grouping variable",
        categorical_cols,
        index=(
            categorical_cols.index("region")
            if "region" in categorical_cols
            else 0
        ),
        key="anova_group"
    )


    anova_metric = st.selectbox(
        "Numerical metric",
        numeric_cols,
        index=(
            numeric_cols.index("charges")
            if "charges" in numeric_cols
            else 0
        ),
        key="anova_metric"
    )


    f_stat, anova_p = anova(
        df,
        anova_group,
        anova_metric
    )


    if f_stat is None:

        st.warning(
            "ANOVA could not be performed."
        )

    else:

        st.write(
            f"**H0:** Mean {anova_metric} is equal "
            f"across all groups of {anova_group}."
        )

        st.write(
            f"**H1:** At least one group has a "
            f"different mean {anova_metric}."
        )


        c1, c2 = st.columns(2)

        c1.metric(
            "F-statistic",
            f"{f_stat:.4f}"
        )

        c2.metric(
            "p-value",
            f"{anova_p:.4g}"
        )


        if anova_p < ALPHA:

            st.success(
                "Reject H0: at least one group mean differs."
            )

        else:

            st.warning(
                "Fail to reject H0: insufficient evidence "
                "that group means differ."
            )


# ============================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header(
        "Live Prediction & Diagnostics"
    )


    # --------------------------------------------------------
    # FIT MODEL
    # --------------------------------------------------------

    model = fit_model(df)


    # --------------------------------------------------------
    # OLS MODEL
    # --------------------------------------------------------

    st.subheader(
        "OLS Model"
    )

    st.code(
        "charges ~ age + bmi + children "
        "+ C(sex) + C(smoker) + C(region)"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "R²",
        f"{model.rsquared:.3f}"
    )

    c2.metric(
        "Adjusted R²",
        f"{model.rsquared_adj:.3f}"
    )

    c3.metric(
        "AIC",
        f"{model.aic:.1f}"
    )

    c4.metric(
        "Observations",
        int(model.nobs)
    )


    # --------------------------------------------------------
    # COEFFICIENT TABLE
    # --------------------------------------------------------

    st.subheader(
        "Regression Coefficients"
    )


    coef_table = pd.DataFrame({
        "Coefficient": model.params,
        "p-value": model.pvalues,
        "CI Lower": model.conf_int()[0],
        "CI Upper": model.conf_int()[1]
    }).round(4)


    st.dataframe(
        coef_table,
        use_container_width=True
    )


    # --------------------------------------------------------
    # LIVE PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "Live Prediction"
    )


    c1, c2, c3 = st.columns(3)


    age = c1.slider(
        "Age",
        18,
        64,
        30
    )


    bmi = c2.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=27.0,
        step=0.1
    )


    children = c3.slider(
        "Children",
        0,
        5,
        0
    )


    c4, c5, c6 = st.columns(3)


    sex = c4.selectbox(
        "Sex",
        sorted(df["sex"].unique())
    )


    smoker = c5.selectbox(
        "Smoker",
        sorted(df["smoker"].unique())
    )


    region = c6.selectbox(
        "Region",
        sorted(df["region"].unique())
    )


    # Create new person's data
    new_person = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [sex],
        "smoker": [smoker],
        "region": [region]
    })


    # Prediction
    prediction = prediction_interval(
        model,
        new_person
    )


    predicted_charge = prediction["mean"].iloc[0]


    st.metric(
        "Predicted Medical Charges",
        f"${predicted_charge:,.2f}"
    )


    # Confidence interval
    st.write(
        "95% Confidence Interval for Mean Response: "
        f"**${prediction['mean_ci_lower'].iloc[0]:,.2f} "
        f"– ${prediction['mean_ci_upper'].iloc[0]:,.2f}**"
    )


    # Prediction interval
    st.write(
        "95% Prediction Interval for an Individual: "
        f"**${prediction['obs_ci_lower'].iloc[0]:,.2f} "
        f"– ${prediction['obs_ci_upper'].iloc[0]:,.2f}**"
    )


    # --------------------------------------------------------
    # RESIDUAL DIAGNOSTICS
    # --------------------------------------------------------

    st.subheader(
        "Residual Diagnostics"
    )


    residuals = model.resid
    fitted = model.fittedvalues


    col1, col2 = st.columns(2)


    # Residual vs fitted
    with col1:

        fig = px.scatter(
            x=fitted,
            y=residuals,
            labels={
                "x": "Fitted Values",
                "y": "Residuals"
            },
            title="Residuals vs Fitted"
        )


        fig.add_hline(
            y=0,
            line_dash="dash"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Q-Q plot
    with col2:

        st.plotly_chart(
            qq_plot(residuals),
            use_container_width=True
        )


    # --------------------------------------------------------
    # NORMALITY TESTS
    # --------------------------------------------------------

    jb = stats.jarque_bera(
        residuals
    )

    omni = sm.stats.omni_normtest(
        residuals
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Jarque-Bera p-value",
        f"{jb.pvalue:.4g}"
    )


    c2.metric(
        "Omnibus p-value",
        f"{omni.pvalue:.4g}"
    )


    # --------------------------------------------------------
    # VIF
    # --------------------------------------------------------

    st.subheader(
        "VIF — Continuous Predictors"
    )


    X = df[
        [
            "age",
            "bmi",
            "children"
        ]
    ]


    X = sm.add_constant(X)


    vif_values = []


    for i in range(1, X.shape[1]):

        vif_values.append(
            variance_inflation_factor(
                X.values,
                i
            )
        )


    vif = pd.DataFrame({
        "Feature": [
            "age",
            "bmi",
            "children"
        ],
        "VIF": vif_values
    })


    st.dataframe(
        vif.round(3),
        use_container_width=True
    )


    st.info(
        "VIF is used to check multicollinearity among "
        "the continuous predictors."
    )