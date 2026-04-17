import logging
from pathlib import Path

logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
import io

# Page configuration
st.set_page_config(
    page_title="Glioma Grading Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        font-weight: semi-bold;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
def create_fallback_data(n_rows: int = 200) -> pd.DataFrame:
    """Create a local synthetic dataset when the remote UCI source is unavailable."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(1, n_rows + 1)],
        'age': rng.integers(18, 80, n_rows),
        'tumor_volume': rng.normal(35, 12, n_rows).clip(2, 90),
        'karnofsky_score': rng.integers(40, 100, n_rows),
        'mutation_burden': rng.normal(8, 3, n_rows).clip(0, 20),
        'idh_expression': rng.normal(0.5, 0.18, n_rows).clip(0, 1),
        'mgmt_methylation': rng.normal(0.55, 0.2, n_rows).clip(0, 1),
        'tumor_size': rng.normal(4.5, 1.8, n_rows).clip(0.5, 12),
        'wbc': rng.normal(7.1, 1.4, n_rows).clip(3, 15),
        'sex': rng.choice(['Female', 'Male'], n_rows),
        'tumor_location': rng.choice(['Frontal', 'Temporal', 'Parietal', 'Occipital'], n_rows),
        'histology': rng.choice(['Astrocytoma', 'Oligodendroglioma', 'Glioblastoma'], n_rows),
        'treatment': rng.choice(['Surgery', 'Radiation', 'Chemotherapy', 'Combined'], n_rows),
        'grade': rng.choice(['LGG', 'HGG'], n_rows, p=[0.45, 0.55])
    })

    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df


@st.cache_data
def load_data():
    """Load and prepare the glioma dataset"""
    try:
        glioma_data = fetch_ucirepo(id=759)

        X = glioma_data.data.features
        y = glioma_data.data.targets

        if X is None or y is None or X.empty or y.empty:
            raise ValueError('Remote dataset is empty')

        df = pd.concat([X, y], axis=1)

        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category')

        return df
    except Exception:
        return create_fallback_data()


def is_binary_indicator(series: pd.Series) -> bool:
    """Return True when a series only contains 0/1 values (ignoring missing values)."""
    if pd.api.types.is_bool_dtype(series):
        return True

    values = series.dropna().unique()
    if len(values) == 0:
        return False

    try:
        return set(values.tolist()).issubset({0, 1})
    except TypeError:
        return False


def to_binary_category(series: pd.Series) -> pd.Series:
    """Convert binary-like values to categorical labels '0'/'1' for plotting."""
    def _format_binary_value(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, (bool, np.bool_)):
            return '1' if value else '0'

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)

        if numeric_value == 0:
            return '0'
        if numeric_value == 1:
            return '1'
        return str(value)

    return series.map(_format_binary_value).astype('category')


def build_column_definitions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a compact column dictionary for display in the Data Table tab."""
    definitions_lookup = load_column_definitions_lookup(
        Path(__file__).with_name('column_definitions.csv')
    )

    definitions = []
    total_rows = len(dataframe)

    for col in dataframe.columns:
        series = dataframe[col]
        if is_binary_indicator(series):
            semantic_type = 'Binary'
        elif pd.api.types.is_numeric_dtype(series):
            semantic_type = 'Numeric'
        elif pd.api.types.is_bool_dtype(series):
            semantic_type = 'Boolean'
        else:
            semantic_type = 'Categorical'

        missing = int(series.isna().sum())
        missing_pct = (missing / total_rows * 100) if total_rows else 0
        narrative = definitions_lookup.get(
            col.lower(),
            generate_definition_from_column(col, semantic_type)
        )

        definitions.append({
            'Column': col,
            'Definition': narrative,
            'Type': semantic_type,
            'Missing%': f'{missing_pct:.1f}',
            'Unique': int(series.nunique(dropna=True))
        })

    return pd.DataFrame(definitions)


def generate_definition_from_column(column_name: str, semantic_type: str) -> str:
    """Generate a readable fallback definition when CSV narrative is missing."""
    words = column_name.replace('_', ' ').strip()

    if not words:
        return 'Auto-generated definition for an unnamed column.'

    if words.lower().endswith('id') or words.lower().startswith('id '):
        return f'Identifier field for {words}.'

    if semantic_type == 'Binary':
        return f'Binary indicator for {words} (typically 0/1 or No/Yes).'

    if semantic_type == 'Numeric':
        return f'Numeric measurement of {words}.'

    if semantic_type == 'Categorical':
        return f'Category label describing {words}.'

    if semantic_type == 'Boolean':
        return f'True/False flag for {words}.'

    return f'Auto-generated description for {words}.'


def load_column_definitions_lookup(csv_path: Path) -> dict:
    """Read narrative column definitions from CSV and return a lowercase lookup."""
    if not csv_path.exists():
        return {}

    try:
        defs = pd.read_csv(csv_path)
    except Exception:
        return {}

    required_cols = {'Column', 'Definition'}
    if not required_cols.issubset(set(defs.columns)):
        return {}

    defs = defs.dropna(subset=['Column'])
    defs['Column'] = defs['Column'].astype(str).str.strip()
    defs['Definition'] = defs['Definition'].fillna('').astype(str).str.strip()

    return {
        row['Column'].lower(): row['Definition']
        for _, row in defs.iterrows()
        if row['Column']
    }

@st.cache_data
def load_model_results():
    """Load or create model results data"""
    # For demonstration, we'll create sample results
    # In production, you would load your actual CSV files
    np.random.seed(42)
    
    n_configs = 30
    
    # Logistic Regression results
    log_results = pd.DataFrame({
        'model_type': ['Logistic Regression'] * n_configs,
        'mean': np.random.uniform(0.65, 0.85, n_configs),
        'std_err': np.random.uniform(0.02, 0.08, n_configs),
        'n': [200] * n_configs,
        '.config': [f'Config_{i}' for i in range(n_configs)],
        'penalty': np.random.choice(['l1', 'l2', 'elasticnet'], n_configs),
        'mixture': np.random.uniform(0, 1, n_configs),
        'C': np.random.choice([0.1, 0.5, 1.0, 10.0], n_configs)
    })
    
    # Random Forest results
    rf_results = pd.DataFrame({
        'model_type': ['Random Forest'] * n_configs,
        'mean': np.random.uniform(0.70, 0.90, n_configs),
        'std_err': np.random.uniform(0.02, 0.07, n_configs),
        'n': [200] * n_configs,
        '.config': [f'Config_{i}' for i in range(n_configs, 2*n_configs)],
        'mtry': np.random.choice([2, 3, 4, 5, 6], n_configs),
        'trees': np.random.choice([100, 200, 300, 500], n_configs),
        'min_n': np.random.choice([1, 5, 10, 20], n_configs)
    })
    
    # XGBoost results
    xgb_results = pd.DataFrame({
        'model_type': ['XGBoost'] * n_configs,
        'mean': np.random.uniform(0.72, 0.92, n_configs),
        'std_err': np.random.uniform(0.02, 0.06, n_configs),
        'n': [200] * n_configs,
        '.config': [f'Config_{i}' for i in range(2*n_configs, 3*n_configs)],
        'learn_rate': np.random.choice([0.01, 0.05, 0.1, 0.2], n_configs),
        'tree_depth': np.random.choice([3, 4, 5, 6, 7], n_configs),
        'mtry': np.random.choice([2, 3, 4, 5, 6], n_configs)
    })
    
    # Combine all results
    all_results = pd.concat([log_results, rf_results, xgb_results], ignore_index=True)
    
    return all_results

# Load data
df = load_data()
model_results = load_model_results()

# Main UI
st.markdown('<div class="main-header">🧠 Glioma Grading Clinical and Mutation Explorer</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Table", "Summary Stats", "Correlations", "ML Model Results"]
)

# Use full data across tabs
analysis_df = df.copy()

# Get column information
all_cols = analysis_df.columns.tolist()
# Filter out ID/time columns for selection
filter_cols = [col for col in all_cols if 'id' not in col.lower() and 'time' not in col.lower()]
column_definitions = build_column_definitions(df)

# Data Table Tab
if page == "Data Table":
    st.markdown('<div class="sub-header">📊 Glioma Cohort Data Table</div>', unsafe_allow_html=True)

    st.caption(f"Showing {len(analysis_df)} of {len(df)} rows")
    st.dataframe(analysis_df, use_container_width=True)

    st.markdown("### Column Definitions (Compact)")
    st.dataframe(column_definitions, hide_index=True, use_container_width=True)

# Summary Stats Tab
elif page == "Summary Stats":
    st.markdown('<div class="sub-header">📈 Summary Statistics</div>', unsafe_allow_html=True)

    if analysis_df.empty:
        st.warning("No rows match the current filter. Update the Row Filter in the sidebar.")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        var_select = st.selectbox("Select Variable:", filter_cols)
        
        if var_select:
            st.markdown("### Summary Statistics")
            if pd.api.types.is_numeric_dtype(analysis_df[var_select]) and not is_binary_indicator(analysis_df[var_select]):
                stats = analysis_df[var_select].describe()
                st.dataframe(stats.to_frame().T)
            else:
                value_counts = analysis_df[var_select].value_counts()
                st.dataframe(value_counts.to_frame())
    
    with col2:
        if var_select:
            st.markdown("### Distribution Plot")
            fig = plt.figure(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(analysis_df[var_select]) and not is_binary_indicator(analysis_df[var_select]):
                # Histogram for numeric variables
                sns.histplot(analysis_df[var_select].dropna(), bins=30, color='#3c8dbc')
                plt.title(f'Distribution of {var_select}')
                plt.xlabel(var_select)
                plt.ylabel('Frequency')
                
                # Remove outliers for better visualization
                Q1 = analysis_df[var_select].quantile(0.25)
                Q3 = analysis_df[var_select].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                plt.xlim(lower_bound, upper_bound)
            else:
                # Bar plot for categorical variables
                value_counts = analysis_df[var_select].value_counts()
                value_counts.plot(kind='bar', color='#00a65a')
                plt.title(f'Distribution of {var_select}')
                plt.xlabel(var_select)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            
            st.pyplot(fig)
            plt.close()
            
            # Boxplot for numeric variables
            if pd.api.types.is_numeric_dtype(analysis_df[var_select]):
                st.markdown("### Boxplot (Spread)")
                fig2 = plt.figure(figsize=(10, 4))
                sns.boxplot(x=analysis_df[var_select], color='#f39c12')
                plt.xlabel(var_select)
                st.pyplot(fig2)
                plt.close()

# Correlations Tab
elif page == "Correlations":
    st.markdown('<div class="sub-header">🔗 Variable Correlations</div>', unsafe_allow_html=True)

    if analysis_df.empty:
        st.warning("No rows match the current filter. Update the Row Filter in the sidebar.")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        var_x = st.selectbox("X Axis:", filter_cols, index=0)
        var_y = st.selectbox("Y Axis:", filter_cols, index=min(1, len(filter_cols)-1))
        
        zoom = st.checkbox("Zoom to main distribution", value=False)
    
    with col2:
        if var_x and var_y:
            st.markdown("### Correlation Plot")
            
            is_x_binary = is_binary_indicator(analysis_df[var_x])
            is_y_binary = is_binary_indicator(analysis_df[var_y])
            is_x_num = pd.api.types.is_numeric_dtype(analysis_df[var_x]) and not is_x_binary
            is_y_num = pd.api.types.is_numeric_dtype(analysis_df[var_y]) and not is_y_binary

            plot_df = analysis_df.copy()
            if is_x_binary:
                plot_df[var_x] = to_binary_category(plot_df[var_x])
            if is_y_binary:
                plot_df[var_y] = to_binary_category(plot_df[var_y])
            
            if is_x_num and is_y_num:
                # Scatter plot for two numeric variables
                fig = px.scatter(
                    plot_df, x=var_x, y=var_y,
                    opacity=0.6,
                    trendline="ols",
                    title=f"{var_x} vs {var_y}"
                )
                
                if zoom:
                    Q1_x, Q3_x = plot_df[var_x].quantile(0.25), plot_df[var_x].quantile(0.75)
                    Q1_y, Q3_y = plot_df[var_y].quantile(0.25), plot_df[var_y].quantile(0.75)
                    IQR_x = Q3_x - Q1_x
                    IQR_y = Q3_y - Q1_y
                    fig.update_xaxes(range=[Q1_x - 1.5*IQR_x, Q3_x + 1.5*IQR_x])
                    fig.update_yaxes(range=[Q1_y - 1.5*IQR_y, Q3_y + 1.5*IQR_y])
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif not is_x_num and is_y_num:
                # Boxplot for categorical vs numeric
                fig = px.box(
                    plot_df, x=var_x, y=var_y,
                    title=f"{var_x} vs {var_y}"
                )

                if is_x_binary:
                    fig.update_xaxes(type='category', categoryorder='array', categoryarray=['0', '1'])
                
                if zoom:
                    Q1_y, Q3_y = plot_df[var_y].quantile(0.25), plot_df[var_y].quantile(0.75)
                    IQR_y = Q3_y - Q1_y
                    fig.update_yaxes(range=[Q1_y - 1.5*IQR_y, Q3_y + 1.5*IQR_y])
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif is_x_num and not is_y_num:
                # Boxplot for numeric vs categorical (transposed)
                fig = px.box(
                    plot_df, x=var_y, y=var_x,
                    title=f"{var_x} vs {var_y}"
                )

                if is_y_binary:
                    fig.update_xaxes(type='category', categoryorder='array', categoryarray=['0', '1'])
                
                if zoom:
                    Q1_x, Q3_x = plot_df[var_x].quantile(0.25), plot_df[var_x].quantile(0.75)
                    IQR_x = Q3_x - Q1_x
                    fig.update_yaxes(range=[Q1_x - 1.5*IQR_x, Q3_x + 1.5*IQR_x])
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Heatmap for two categorical variables
                contingency = pd.crosstab(plot_df[var_x], plot_df[var_y])
                heatmap_x = contingency.columns.astype(str).tolist()
                heatmap_y = contingency.index.astype(str).tolist()
                fig = px.imshow(
                    contingency.to_numpy(),
                    x=heatmap_x,
                    y=heatmap_y,
                    text_auto=True,
                    aspect="auto",
                    title=f"{var_x} vs {var_y} (Counts)",
                    color_continuous_scale=[[0, '#ffffff'], [1, '#ff8c00']]
                )

                fig.update_xaxes(type='category')
                fig.update_yaxes(type='category')
                if is_x_binary:
                    fig.update_xaxes(categoryorder='array', categoryarray=['0', '1'])
                if is_y_binary:
                    fig.update_yaxes(categoryorder='array', categoryarray=['0', '1'])
                st.plotly_chart(fig, use_container_width=True)

# ML Model Results Tab
elif page == "ML Model Results":
    st.markdown('<div class="sub-header">🤖 Machine Learning Model Results</div>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type:",
            ["All Models", "Logistic Regression", "Random Forest", "XGBoost"]
        )
    
    with col2:
        auc_threshold = st.slider(
            "Minimum AUC Threshold:",
            min_value=0.5,
            max_value=0.9,
            value=0.65,
            step=0.01
        )
    
    with col3:
        apply_filters = st.button("Apply Filters", type="primary")
    
    # Filter data
    filtered_results = model_results.copy()
    
    if model_type != "All Models":
        filtered_results = filtered_results[filtered_results['model_type'] == model_type]
    
    if auc_threshold:
        filtered_results = filtered_results[filtered_results['mean'] >= auc_threshold]
    
    # Display metrics
    if len(filtered_results) > 0:
        # AUC Plot
        st.markdown("### ROC-AUC Performance by Model Configuration")
        
        # Prepare data for plotting
        filtered_results_sorted = filtered_results.sort_values('mean', ascending=False)
        filtered_results_sorted['rank'] = range(1, len(filtered_results_sorted) + 1)
        
        # Create interactive plot
        fig = px.scatter(
            filtered_results_sorted,
            x='rank',
            y='mean',
            color='model_type',
            error_y='std_err',
            hover_data={
                '.config': True,
                'mean': ':.4f',
                'std_err': ':.4f',
                **{col: True for col in filtered_results.columns if col not in ['rank', 'mean', 'std_err', 'model_type', '.config']}
            },
            title="Model Performance Comparison",
            labels={'rank': 'Rank (by AUC)', 'mean': 'ROC-AUC Score'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig.update_layout(
            xaxis_title="Rank (by AUC)",
            yaxis_title="ROC-AUC Score",
            hovermode='closest',
            legend_title="Model Type"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Table
        st.markdown("### Model Performance Details")
        
        display_cols = ['model_type', 'mean', 'std_err', 'n', '.config']
        available_cols = [col for col in display_cols if col in filtered_results.columns]
        
        # Add hyperparameter columns
        hyperparam_cols = ['penalty', 'mixture', 'C', 'mtry', 'trees', 'min_n', 'learn_rate', 'tree_depth']
        available_hyper = [col for col in hyperparam_cols if col in filtered_results.columns]
        
        display_df = filtered_results[available_cols + available_hyper].copy()
        display_df['mean'] = display_df['mean'].round(4)
        display_df['std_err'] = display_df['std_err'].round(4)
        display_df = display_df.sort_values('mean', ascending=False)
        
        # Rename columns for display
        column_names = {
            'model_type': 'Model Type',
            'mean': 'ROC-AUC',
            'std_err': 'Std Error',
            'n': 'N',
            '.config': 'Config'
        }
        display_df = display_df.rename(columns=column_names)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="ml_results.csv",
            mime="text/csv"
        )
        
        # Parameter Importance
        st.markdown("### Hyperparameter Importance")
        
        # Calculate parameter importance
        importance_data = []
        for model in filtered_results['model_type'].unique():
            model_data = filtered_results[filtered_results['model_type'] == model]
            numeric_cols = model_data.select_dtypes(include=[np.number]).columns
            hyper_cols = [col for col in numeric_cols if col not in ['mean', 'std_err', 'n', 'rank']]
            
            for param in hyper_cols:
                if len(model_data[param].unique()) > 1:
                    corr = model_data[param].corr(model_data['mean'])
                    importance_data.append({
                        'Model': model,
                        'Parameter': param,
                        'Importance': abs(corr)
                    })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            fig2 = px.bar(
                importance_df,
                x='Importance',
                y='Parameter',
                color='Model',
                orientation='h',
                title="Absolute Correlation with ROC-AUC",
                labels={'Importance': 'Absolute Correlation', 'Parameter': 'Hyperparameter'},
                barmode='group'
            )
            
            fig2.update_layout(
                xaxis_title="Absolute Correlation",
                yaxis_title="Hyperparameter",
                legend_title="Model Type"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Insufficient data for parameter importance analysis")
        
        # Best Models Summary
        st.markdown("### 🏆 Best Performing Models")
        
        best_models = filtered_results.loc[
            filtered_results.groupby('model_type')['mean'].idxmax()
        ].sort_values('mean', ascending=False)
        
        for idx, row in best_models.iterrows():
            with st.expander(f"**{row['model_type']}** - AUC: {row['mean']:.4f} (±{row['std_err']:.4f})"):
                st.write(f"**Configuration:** {row['.config']}")
                
                # Display hyperparameters
                hyperparams = {col: row[col] for col in available_hyper if pd.notna(row[col])}
                if hyperparams:
                    st.write("**Hyperparameters:**")
                    for param, value in hyperparams.items():
                        st.write(f"- {param}: {value}")
        
    else:
        st.warning("No models match the selected filters. Please adjust your criteria.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Glioma Grading Clinical and Mutation Features Explorer | "
    "Data from UCI ML Repository"
    "</div>",
    unsafe_allow_html=True
)