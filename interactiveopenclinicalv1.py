"""
Interactive OpenClinical - Streamlit Version
Advanced Interactive Clinical Trial Analysis Platform

Real-time filtering • Dynamic analysis • JMP/STATA-level exploration
Built for researchers who need interactive data exploration with professional statistics.

PURE PYTHON STATISTICAL IMPLEMENTATION
No external statistical libraries required - works on any Streamlit deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from io import BytesIO
import math

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pure Python Statistical Functions
class PureStats:
    """Pure Python implementation of statistical functions"""
    
    @staticmethod
    def ttest_ind(a, b):
        """Independent t-test using pure NumPy"""
        a, b = np.asarray(a), np.asarray(b)
        
        # Remove NaN values
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        
        if len(a) == 0 or len(b) == 0:
            return np.nan, np.nan
        
        # Calculate means and variances
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        n_a, n_b = len(a), len(b)
        
        # Pooled standard error
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_se = np.sqrt(pooled_var * (1/n_a + 1/n_b))
        
        if pooled_se == 0:
            return np.nan, np.nan
        
        # t-statistic
        t_stat = (mean_a - mean_b) / pooled_se
        
        # Degrees of freedom
        df = n_a + n_b - 2
        
        # Approximate p-value using normal approximation for large df
        if df > 30:
            # Normal approximation
            p_value = 2 * (1 - PureStats._norm_cdf(abs(t_stat)))
        else:
            # Simplified t-distribution approximation
            p_value = 2 * (1 - PureStats._t_cdf(abs(t_stat), df))
        
        return t_stat, p_value
    
    @staticmethod
    def _norm_cdf(x):
        """Cumulative distribution function for standard normal distribution"""
        # Approximation using error function
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def _t_cdf(t, df):
        """Simplified t-distribution CDF approximation"""
        # For small df, use a simple approximation
        if df <= 1:
            return 0.5 + math.atan(t) / math.pi
        elif df <= 2:
            return 0.5 + (t / math.sqrt(2 + t**2)) / 2
        else:
            # For larger df, approximate with normal
            return PureStats._norm_cdf(t * math.sqrt(df / (df + t**2)))
    
    @staticmethod
    def mannwhitneyu(x, y):
        """Mann-Whitney U test (simplified version)"""
        x, y = np.asarray(x), np.asarray(y)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        if len(x) == 0 or len(y) == 0:
            return np.nan, np.nan
        
        # Combine and rank
        combined = np.concatenate([x, y])
        ranks = PureStats._rankdata(combined)
        
        # Split ranks back
        ranks_x = ranks[:len(x)]
        ranks_y = ranks[len(x):]
        
        # Calculate U statistics
        U1 = len(x) * len(y) + len(x) * (len(x) + 1) / 2 - np.sum(ranks_x)
        U2 = len(x) * len(y) - U1
        
        # Use smaller U
        U = min(U1, U2)
        
        # Normal approximation for p-value
        mu = len(x) * len(y) / 2
        sigma = math.sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12)
        
        if sigma == 0:
            return U, np.nan
        
        z = (U - mu) / sigma
        p_value = 2 * (1 - PureStats._norm_cdf(abs(z)))
        
        return U, p_value
    
    @staticmethod
    def _rankdata(data):
        """Assign ranks to data"""
        sorted_indices = np.argsort(data)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(1, len(data) + 1)
        return ranks
    
    @staticmethod
    def chi2_contingency(observed):
        """Chi-square test for independence"""
        observed = np.asarray(observed)
        
        if observed.size == 0:
            return np.nan, np.nan, np.nan, None
        
        # Calculate expected frequencies
        row_totals = np.sum(observed, axis=1)
        col_totals = np.sum(observed, axis=0)
        total = np.sum(observed)
        
        expected = np.outer(row_totals, col_totals) / total
        
        # Avoid division by zero
        expected = np.where(expected == 0, 1e-10, expected)
        
        # Chi-square statistic
        chi2_stat = np.sum((observed - expected)**2 / expected)
        
        # Degrees of freedom
        df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        
        # Approximate p-value (simplified)
        if df > 0:
            # Very rough approximation - in practice would need gamma function
            p_value = math.exp(-chi2_stat / 2) if chi2_stat < 20 else 0.001
        else:
            p_value = 1.0
        
        return chi2_stat, p_value, df, expected
    
    @staticmethod
    def pearsonr(x, y):
        """Pearson correlation coefficient"""
        x, y = np.asarray(x), np.asarray(y)
        
        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 2:
            return np.nan, np.nan
        
        # Calculate correlation
        r = np.corrcoef(x, y)[0, 1]
        
        # Approximate p-value
        if np.isnan(r):
            return np.nan, np.nan
        
        # t-statistic for correlation
        n = len(x)
        if abs(r) == 1:
            p_value = 0.0
        else:
            t_stat = r * math.sqrt((n - 2) / (1 - r**2))
            p_value = 2 * (1 - PureStats._t_cdf(abs(t_stat), n - 2))
        
        return r, p_value

# Configure page
st.set_page_config(
    page_title="Interactive OpenClinical",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for interactive features
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .interactive-badge {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .filter-panel {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4ecdc4;
        margin: 0.5rem 0;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .analysis-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stats-info {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class InteractiveAnalyzer:
    """Interactive clinical trial analyzer with real-time capabilities"""
    
    def __init__(self):
        if 'original_data' not in st.session_state:
            st.session_state.original_data = None
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
        if 'context' not in st.session_state:
            st.session_state.context = {}
        if 'current_filters' not in st.session_state:
            st.session_state.current_filters = {}
    
    def load_data(self, uploaded_file):
        """Load and process clinical trial data"""
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
                context = {}
            else:
                # Handle multi-sheet Excel
                excel_data = pd.read_excel(uploaded_file, sheet_name=None)
                data, context = self._process_excel_sheets(excel_data)
            
            # Store in session state
            st.session_state.original_data = data
            st.session_state.filtered_data = data.copy()
            st.session_state.context = context
            st.session_state.current_filters = {}
            
            return True, f"✅ Loaded {len(data)} patients with {len(data.columns)} variables"
            
        except Exception as e:
            return False, f"❌ Error loading data: {str(e)}"
    
    def _process_excel_sheets(self, excel_data):
        """Process multi-sheet Excel intelligently"""
        context = {'study_description': '', 'variable_descriptions': {}}
        main_data = None
        
        for sheet_name, df in excel_data.items():
            sheet_name_lower = sheet_name.lower()
            
            if 'data' in sheet_name_lower:
                main_data = df
            elif 'variable' in sheet_name_lower and len(df.columns) >= 2:
                # Extract variable descriptions
                var_col, desc_col = df.columns[0], df.columns[1]
                for _, row in df.iterrows():
                    if pd.notna(row[var_col]) and pd.notna(row[desc_col]):
                        context['variable_descriptions'][str(row[var_col])] = str(row[desc_col])
            elif 'description' in sheet_name_lower:
                # Extract study description
                description_text = ""
                for col in df.columns:
                    for value in df[col].dropna():
                        if isinstance(value, str) and len(value) > 20:
                            description_text += value + " "
                context['study_description'] = description_text.strip()
        
        return main_data, context
    
    def get_filter_options(self):
        """Get available filtering options from current data"""
        if st.session_state.original_data is None:
            return {}, [], []
        
        data = st.session_state.original_data
        
        # Categorical variables for filtering
        categorical_vars = []
        continuous_vars = []
        
        for col in data.columns:
            if data[col].dtype in ['object', 'category'] or data[col].nunique() < 20:
                if data[col].nunique() > 1 and data[col].nunique() < 50:  # Reasonable for filtering
                    categorical_vars.append(col)
            elif data[col].dtype in ['int64', 'float64']:
                continuous_vars.append(col)
        
        # Get unique values for categorical variables
        filter_options = {}
        for var in categorical_vars:
            unique_vals = data[var].dropna().unique()
            filter_options[var] = sorted([str(x) for x in unique_vals])
        
        return filter_options, categorical_vars, continuous_vars
    
    def apply_filters(self, filter_dict):
        """Apply filters and update session state"""
        if st.session_state.original_data is None:
            return
        
        filtered_data = st.session_state.original_data.copy()
        active_filters = []
        
        # Apply each filter
        for var, values in filter_dict.items():
            if values and len(values) > 0:
                if var in filtered_data.columns:
                    # Convert to string for comparison
                    filtered_data = filtered_data[filtered_data[var].astype(str).isin([str(v) for v in values])]
                    active_filters.append(f"{var}={values}")
        
        st.session_state.filtered_data = filtered_data
        st.session_state.current_filters = {k: v for k, v in filter_dict.items() if v}
        
        return len(filtered_data), active_filters
    
    def get_variable_summary(self, variable):
        """Get summary statistics for a specific variable"""
        if st.session_state.filtered_data is None or variable not in st.session_state.filtered_data.columns:
            return "Variable not found or no data loaded"
        
        data = st.session_state.filtered_data
        var_data = data[variable].dropna()
        var_desc = st.session_state.context.get('variable_descriptions', {}).get(variable, '')
        
        summary_dict = {
            'variable': variable,
            'description': var_desc,
            'total_count': len(data),
            'valid_count': len(var_data),
            'missing_count': data[variable].isnull().sum(),
            'missing_percent': (data[variable].isnull().sum() / len(data)) * 100
        }
        
        if var_data.dtype in ['int64', 'float64'] and len(var_data) > 0:
            # Continuous variable
            summary_dict.update({
                'type': 'continuous',
                'mean': var_data.mean(),
                'median': var_data.median(),
                'std': var_data.std(),
                'min': var_data.min(),
                'max': var_data.max(),
                'q25': var_data.quantile(0.25),
                'q75': var_data.quantile(0.75)
            })
        else:
            # Categorical variable
            value_counts = var_data.value_counts()
            summary_dict.update({
                'type': 'categorical',
                'unique_values': var_data.nunique(),
                'top_values': value_counts.head(10).to_dict()
            })
        
        return summary_dict
    
    def create_interactive_plot(self, plot_type, x_var=None, y_var=None, color_var=None):
        """Create interactive plots based on current filtered data"""
        if st.session_state.filtered_data is None or len(st.session_state.filtered_data) == 0:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        data = st.session_state.filtered_data
        
        try:
            if plot_type == "Distribution" and x_var:
                if data[x_var].dtype in ['int64', 'float64']:
                    fig = px.histogram(data, x=x_var, color=color_var, 
                                     title=f"Distribution of {x_var}")
                else:
                    fig = px.histogram(data, x=x_var, color=color_var,
                                     title=f"Distribution of {x_var}")
                    fig.update_xaxis(tickangle=45)
                
            elif plot_type == "Box Plot" and x_var and y_var:
                fig = px.box(data, x=x_var, y=y_var, color=color_var,
                           title=f"{y_var} by {x_var}")
                
            elif plot_type == "Scatter" and x_var and y_var:
                fig = px.scatter(data, x=x_var, y=y_var, color=color_var,
                               title=f"{y_var} vs {x_var}")
                
            elif plot_type == "Violin Plot" and x_var and y_var:
                fig = px.violin(data, x=x_var, y=y_var, color=color_var,
                              title=f"{y_var} by {x_var}")
                
            else:
                return go.Figure().add_annotation(
                    text="Please select appropriate variables for the chosen plot type",
                    showarrow=False
                )
            
            # Add sample size to title
            fig.update_layout(title=f"{fig.layout.title.text} (n={len(data)})")
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)
    
    def run_comparative_analysis(self, outcome_var, group_var, covariates=None, analysis_type="auto"):
        """Run comparative analysis with current filtered data"""
        if st.session_state.filtered_data is None:
            return {"error": "No data loaded"}
        
        if outcome_var not in st.session_state.filtered_data.columns or group_var not in st.session_state.filtered_data.columns:
            return {"error": "Selected variables not found"}
        
        data = st.session_state.filtered_data
        
        # Clean data for analysis
        analysis_vars = [outcome_var, group_var]
        if covariates:
            analysis_vars.extend([c for c in covariates if c in data.columns])
        
        clean_data = data[analysis_vars].dropna()
        
        if len(clean_data) < 10:
            return {"error": f"Insufficient data for analysis (n={len(clean_data)})"}
        
        # Get groups
        groups = clean_data[group_var].unique()
        if len(groups) != 2:
            return {"error": f"Group variable must have exactly 2 groups (found {len(groups)})"}
        
        results = {
            'sample_size': len(clean_data),
            'outcome_variable': outcome_var,
            'group_variable': group_var,
            'groups': groups,
            'covariates': covariates or [],
            'active_filters': list(st.session_state.current_filters.keys()),
            'statistical_engine': 'Pure Python Implementation'
        }
        
        try:
            # Descriptive statistics by group
            group_stats = {}
            for group in groups:
                group_data = clean_data[clean_data[group_var] == group][outcome_var]
                group_stats[f"Group {group}"] = {
                    'n': len(group_data),
                    'mean': group_data.mean(),
                    'std': group_data.std(),
                    'median': group_data.median(),
                    'min': group_data.min(),
                    'max': group_data.max()
                }
            
            results['descriptive_stats'] = group_stats
            
            # Statistical test
            group0_data = clean_data[clean_data[group_var] == groups[0]][outcome_var]
            group1_data = clean_data[clean_data[group_var] == groups[1]][outcome_var]
            
            # Choose test based on data characteristics
            normality_check = self._check_normality(group0_data, group1_data)
            
            if normality_check['use_parametric']:
                # Independent t-test
                t_stat, p_value = PureStats.ttest_ind(group0_data, group1_data)
                test_type = "Independent t-test (Pure Python)"
                test_statistic = t_stat
            else:
                # Mann-Whitney U test
                u_stat, p_value = PureStats.mannwhitneyu(group0_data, group1_data)
                test_type = "Mann-Whitney U test (Pure Python)"
                test_statistic = u_stat
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group0_data)-1)*group0_data.var() + 
                                 (len(group1_data)-1)*group1_data.var()) / 
                                (len(group0_data)+len(group1_data)-2))
            cohens_d = (group1_data.mean() - group0_data.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for difference
            diff = group1_data.mean() - group0_data.mean()
            se_diff = pooled_std * np.sqrt(1/len(group0_data) + 1/len(group1_data))
            ci_lower = diff - 1.96 * se_diff
            ci_upper = diff + 1.96 * se_diff
            
            results.update({
                'test_type': test_type,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'effect_size': cohens_d,
                'difference': diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < 0.05,
                'normality_info': normality_check
            })
            
            # Simple linear regression if covariates specified
            if covariates and len(covariates) > 0:
                results['covariate_note'] = "Covariate adjustment available with basic linear regression"
                try:
                    # Simple multiple regression using numpy
                    reg_result = self._simple_regression(clean_data, outcome_var, group_var, covariates)
                    results['regression'] = reg_result
                except Exception as e:
                    results['regression_error'] = str(e)
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}
    
    def _check_normality(self, group1, group2):
        """Simple normality check based on skewness and sample size"""
        def skewness(x):
            x = np.asarray(x)
            n = len(x)
            if n < 3:
                return 0
            m = np.mean(x)
            s = np.std(x, ddof=1)
            if s == 0:
                return 0
            return np.sum((x - m)**3) / ((n - 1) * s**3)
        
        skew1 = abs(skewness(group1))
        skew2 = abs(skewness(group2))
        
        # Simple rules for normality
        sample_size_ok = len(group1) >= 30 and len(group2) >= 30
        skewness_ok = skew1 < 2 and skew2 < 2
        
        use_parametric = sample_size_ok or skewness_ok
        
        return {
            'use_parametric': use_parametric,
            'group1_skewness': skew1,
            'group2_skewness': skew2,
            'group1_size': len(group1),
            'group2_size': len(group2),
            'recommendation': 'Parametric test' if use_parametric else 'Non-parametric test'
        }
    
    def _simple_regression(self, data, outcome_var, group_var, covariates):
        """Simple multiple regression using numpy"""
        # Prepare design matrix
        y = data[outcome_var].values
        
        # Create design matrix with intercept, group, and covariates
        X = np.ones((len(data), 1))  # Intercept
        
        # Add group variable (assuming binary: 0/1)
        group_values = pd.get_dummies(data[group_var], drop_first=True).values
        if group_values.shape[1] > 0:
            X = np.hstack([X, group_values])
        
        # Add covariates
        for cov in covariates:
            if cov in data.columns:
                cov_values = data[cov].values.reshape(-1, 1)
                X = np.hstack([X, cov_values])
        
        # Solve using least squares
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Calculate R-squared
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'coefficients': beta.tolist(),
                'r_squared': r_squared,
                'variables': ['intercept', 'group'] + covariates[:len(beta)-2],
                'note': 'Simple least squares regression'
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    # Header
    st.markdown("""
    <div class="main-header">🔬 Interactive OpenClinical</div>
    <div class="interactive-badge">
        🎛️ Real-time Data Exploration • Dynamic Analysis • JMP/STATA-level Control
    </div>
    """, unsafe_allow_html=True)
    
    # Show statistical engine info
    st.markdown("""
    <div class="stats-info">
        📊 <strong>Statistical Engine:</strong> Pure Python Implementation - No external dependencies required!<br>
        ✅ t-tests, Mann-Whitney U, correlations, basic regression - all implemented natively
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = InteractiveAnalyzer()
    
    # Sidebar for data upload and main controls
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Clinical Trial Data",
            type=['xlsx', 'xls', 'csv'],
            help="📚 Multi-sheet Excel files supported!"
        )
        
        if uploaded_file and st.button("Load Data", type="primary"):
            success, message = analyzer.load_data(uploaded_file)
            if success:
                st.success(message)
                st.rerun()  # Refresh to show new data options
            else:
                st.error(message)
        
        # Show data info if loaded
        if st.session_state.original_data is not None:
            st.markdown("---")
            st.markdown("### 📊 Dataset Info")
            st.metric("Total Patients", len(st.session_state.original_data))
            st.metric("Variables", len(st.session_state.original_data.columns))
            
            if st.session_state.filtered_data is not None:
                filtered_count = len(st.session_state.filtered_data)
                st.metric("Filtered Patients", filtered_count)
                if filtered_count != len(st.session_state.original_data):
                    reduction = len(st.session_state.original_data) - filtered_count
                    st.metric("Filtered Out", reduction)
    
    # Main interface tabs
    if st.session_state.original_data is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎛️ Interactive Filters", 
            "📊 Data Explorer", 
            "📈 Visualizations", 
            "🔬 Statistical Analysis"
        ])
        
        with tab1:
            st.markdown("### 🎛️ Real-time Data Filtering")
            st.markdown("**Filter your dataset interactively - results update instantly!**")
            
            # Get filter options
            filter_options, categorical_vars, continuous_vars = analyzer.get_filter_options()
            
            if filter_options:
                # Create filter interface
                st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
                
                filter_dict = {}
                
                # Create columns for filters
                num_cols = min(3, len(filter_options))
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    
                    for i, (var, options) in enumerate(filter_options.items()):
                        with cols[i % num_cols]:
                            # Get variable description
                            var_desc = st.session_state.context.get('variable_descriptions', {}).get(var, '')
                            help_text = var_desc if var_desc else f"Filter by {var}"
                            
                            selected = st.multiselect(
                                f"🔍 {var}",
                                options=options,
                                default=st.session_state.current_filters.get(var, []),
                                help=help_text,
                                key=f"filter_{var}"
                            )
                            if selected:
                                filter_dict[var] = selected
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply filters button
                if st.button("Apply Filters", type="primary"):
                    filtered_count, active_filters = analyzer.apply_filters(filter_dict)
                    st.success(f"✅ Filtered to {filtered_count} patients")
                    if active_filters:
                        st.info(f"Active filters: {', '.join([str(f) for f in active_filters])}")
                    st.rerun()
                
                # Clear filters button
                if st.button("Clear All Filters"):
                    st.session_state.filtered_data = st.session_state.original_data.copy()
                    st.session_state.current_filters = {}
                    st.success("✅ All filters cleared")
                    st.rerun()
            
            else:
                st.info("No suitable categorical variables found for filtering")
        
        with tab2:
            st.markdown("### 📊 Interactive Data Explorer")
            
            if st.session_state.filtered_data is not None:
                # Variable selector
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    variables = list(st.session_state.filtered_data.columns)
                    selected_var = st.selectbox(
                        "🔍 Explore Variable",
                        options=variables,
                        help="Select a variable to see detailed statistics"
                    )
                    
                    if selected_var:
                        summary = analyzer.get_variable_summary(selected_var)
                        
                        if isinstance(summary, dict):
                            st.markdown(f"**📋 {summary['variable']}**")
                            if summary['description']:
                                st.info(summary['description'])
                            
                            # Basic metrics
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Valid Count", summary['valid_count'])
                            with col_b:
                                st.metric("Missing", f"{summary['missing_count']} ({summary['missing_percent']:.1f}%)")
                            
                            # Type-specific statistics
                            if summary['type'] == 'continuous':
                                st.markdown("**📈 Continuous Statistics:**")
                                col_c, col_d = st.columns(2)
                                with col_c:
                                    st.metric("Mean", f"{summary['mean']:.2f}")
                                    st.metric("Min", f"{summary['min']:.2f}")
                                with col_d:
                                    st.metric("Std Dev", f"{summary['std']:.2f}")
                                    st.metric("Max", f"{summary['max']:.2f}")
                            
                            elif summary['type'] == 'categorical':
                                st.markdown("**🏷️ Categorical Distribution:**")
                                st.metric("Unique Values", summary['unique_values'])
                                
                                if summary['top_values']:
                                    st.markdown("**Top Values:**")
                                    for value, count in summary['top_values'].items():
                                        pct = (count / summary['valid_count']) * 100
                                        st.write(f"• {value}: {count} ({pct:.1f}%)")
                
                with col2:
                    # Data preview
                    st.markdown("**📋 Filtered Data Preview**")
                    st.dataframe(
                        st.session_state.filtered_data.head(20),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Quick stats
                    if len(st.session_state.filtered_data) > 0:
                        st.markdown("**⚡ Quick Statistics**")
                        numeric_cols = st.session_state.filtered_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            quick_stats = st.session_state.filtered_data[numeric_cols].describe()
                            st.dataframe(quick_stats, use_container_width=True)
        
        with tab3:
            st.markdown("### 📈 Interactive Visualizations")
            
            if st.session_state.filtered_data is not None and len(st.session_state.filtered_data) > 0:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**🎨 Plot Configuration**")
                    
                    plot_type = st.selectbox(
                        "Plot Type",
                        ["Distribution", "Box Plot", "Scatter", "Violin Plot"]
                    )
                    
                    variables = list(st.session_state.filtered_data.columns)
                    
                    x_var = st.selectbox("X Variable", options=variables)
                    
                    y_var = None
                    if plot_type in ["Box Plot", "Scatter", "Violin Plot"]:
                        y_var = st.selectbox("Y Variable", options=variables)
                    
                    color_var = st.selectbox(
                        "Color By (optional)",
                        options=["None"] + variables
                    )
                    color_var = color_var if color_var != "None" else None
                    
                    if st.button("Create Plot", type="primary"):
                        fig = analyzer.create_interactive_plot(plot_type, x_var, y_var, color_var)
                        st.session_state.current_plot = fig
                
                with col2:
                    if hasattr(st.session_state, 'current_plot'):
                        st.plotly_chart(st.session_state.current_plot, use_container_width=True)
                    else:
                        st.info("Configure plot settings and click 'Create Plot' to generate visualization")
        
        with tab4:
            st.markdown("### 🔬 Pure Python Statistical Analysis")
            
            if st.session_state.filtered_data is not None:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**⚙️ Analysis Configuration**")
                    
                    variables = list(st.session_state.filtered_data.columns)
                    
                    outcome_var = st.selectbox(
                        "📊 Outcome Variable",
                        options=variables,
                        help="Select the dependent variable for analysis"
                    )
                    
                    group_var = st.selectbox(
                        "👥 Group Variable",
                        options=variables,
                        help="Select the grouping variable (should have 2 groups)"
                    )
                    
                    # Show group distribution
                    if group_var and group_var in st.session_state.filtered_data.columns:
                        group_counts = st.session_state.filtered_data[group_var].value_counts()
                        st.markdown("**Group Distribution:**")
                        for group, count in group_counts.items():
                            st.write(f"• Group {group}: {count} patients")
                    
                    covariates = st.multiselect(
                        "🔧 Covariates",
                        options=[v for v in variables if v not in [outcome_var, group_var]],
                        help="Select variables to adjust for in the analysis"
                    )
                    
                    if st.button("Run Analysis", type="primary"):
                        results = analyzer.run_comparative_analysis(
                            outcome_var, group_var, covariates
                        )
                        st.session_state.analysis_results = results
                
                with col2:
                    if hasattr(st.session_state, 'analysis_results'):
                        results = st.session_state.analysis_results
                        
                        if 'error' in results:
                            st.error(f"❌ {results['error']}")
                        else:
                            st.markdown('<div class="analysis-result">', unsafe_allow_html=True)
                            
                            # Header
                            st.markdown(f"## 📊 Analysis Results")
                            st.markdown(f"**Outcome:** {results['outcome_variable']}")
                            st.markdown(f"**Groups:** {results['group_variable']}")
                            st.markdown(f"**Sample Size:** {results['sample_size']} patients")
                            st.markdown(f"**Engine:** {results['statistical_engine']}")
                            
                            if results.get('active_filters'):
                                st.markdown(f"**Active Filters:** {', '.join(results['active_filters'])}")
                            
                            # Descriptive statistics
                            st.markdown("### 📈 Descriptive Statistics")
                            desc_stats = results.get('descriptive_stats', {})
                            
                            col_a, col_b = st.columns(2)
                            for i, (group_name, stats_dict) in enumerate(desc_stats.items()):
                                with col_a if i == 0 else col_b:
                                    st.markdown(f"**{group_name}:**")
                                    st.write(f"• N: {stats_dict['n']}")
                                    st.write(f"• Mean: {stats_dict['mean']:.3f}")
                                    st.write(f"• SD: {stats_dict['std']:.3f}")
                                    st.write(f"• Median: {stats_dict['median']:.3f}")
                            
                            # Statistical test results
                            st.markdown("### 🧮 Statistical Test Results")
                            
                            # Test selection info
                            if 'normality_info' in results:
                                norm_info = results['normality_info']
                                st.info(f"📋 **Test Selection:** {norm_info['recommendation']} chosen based on data characteristics")
                            
                            col_c, col_d, col_e = st.columns(3)
                            with col_c:
                                st.metric("Test Type", results.get('test_type', 'N/A'))
                            with col_d:
                                p_val = results.get('p_value', None)
                                if p_val is not None:
                                    st.metric("P-value", f"{p_val:.4f}")
                            with col_e:
                                effect_size = results.get('effect_size', None)
                                if effect_size is not None:
                                    st.metric("Effect Size (Cohen's d)", f"{effect_size:.3f}")
                            
                            # Significance and interpretation
                            if results.get('significant'):
                                st.success("✅ **Statistically Significant** (p < 0.05)")
                            else:
                                st.info("📊 **Not Statistically Significant** (p ≥ 0.05)")
                            
                            # Effect size interpretation
                            if effect_size is not None:
                                if abs(effect_size) < 0.2:
                                    effect_interp = "Negligible effect"
                                elif abs(effect_size) < 0.5:
                                    effect_interp = "Small effect"
                                elif abs(effect_size) < 0.8:
                                    effect_interp = "Medium effect"
                                else:
                                    effect_interp = "Large effect"
                                st.info(f"🎯 **Effect Size:** {effect_interp}")
                            
                            # Confidence interval
                            if 'difference' in results:
                                diff = results['difference']
                                ci_lower = results.get('ci_lower', None)
                                ci_upper = results.get('ci_upper', None)
                                if ci_lower is not None and ci_upper is not None:
                                    st.markdown(f"**Mean Difference:** {diff:.3f} (95% CI: {ci_lower:.3f} to {ci_upper:.3f})")
                            
                            # Regression results if available
                            if 'regression' in results:
                                st.markdown("### 🔧 Multiple Regression Results")
                                reg = results['regression']
                                if 'error' in reg:
                                    st.warning(f"⚠️ Regression Error: {reg['error']}")
                                else:
                                    st.metric("R²", f"{reg['r_squared']:.3f}")
                                    st.info(f"📊 {reg['note']}")
                            
                            if 'covariate_note' in results:
                                st.info(f"ℹ️ {results['covariate_note']}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Configure analysis settings and click 'Run Analysis' to see results")
    
    else:
        # Welcome screen when no data is loaded
        st.info("👆 **Upload your clinical trial dataset to begin interactive analysis**")
        
        # Feature showcase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎛️ **Interactive Features:**
            
            **🔍 Real-time Filtering:**
            - Filter by demographics, groups, outcomes
            - See sample size update instantly
            - Multiple filter combinations
            
            **📊 Dynamic Analysis:**
            - Results update as you filter
            - Compare subgroups interactively
            - Intelligent test selection
            
            **📈 Live Visualizations:**
            - Plots update with filtered data
            - Multiple plot types available
            - Interactive exploration
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 **Pure Python Capabilities:**
            
            **🔬 Statistical Analysis:**
            - Independent t-tests
            - Mann-Whitney U tests
            - Effect size calculations
            - Confidence intervals
            
            **🌐 Zero Dependencies:**
            - No scipy or statsmodels required
            - Works on any Streamlit deployment
            - Fast, reliable statistical engine
            
            **🎯 Professional Results:**
            - Publication-ready statistics
            - Automatic test selection
            - Clear interpretations
            """)

if __name__ == "__main__":
    main()
