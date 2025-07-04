"""
Interactive OpenClinical - Streamlit Version
Advanced Interactive Clinical Trial Analysis Platform

Real-time filtering • Dynamic analysis • JMP/STATA-level exploration
Built for researchers who need interactive data exploration with professional statistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, fisher_exact

# Compatibility fix for statsmodels/scipy issue
try:
    # Try to import statsmodels normally first
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
except ImportError as e:
    if "_lazywhere" in str(e):
        # Apply the compatibility workaround
        import scipy
        from numpy import where as _lazywhere
        # Monkey patch the missing function
        if not hasattr(scipy._lib._util, '_lazywhere'):
            scipy._lib._util._lazywhere = _lazywhere
        
        # Now try importing again
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        st.warning("⚠️ Applied compatibility fix for statsmodels/scipy versions")
    else:
        # Different import error, re-raise
        raise

from datetime import datetime
import warnings
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from io import BytesIO

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            'active_filters': list(st.session_state.current_filters.keys())
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
            
            # Independent t-test
            t_stat, p_value = ttest_ind(group0_data, group1_data)
            
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
                'test_type': 'Independent t-test',
                'test_statistic': t_stat,
                'p_value': p_value,
                'effect_size': cohens_d,
                'difference': diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < 0.05
            })
            
            # ANCOVA if covariates specified
            if covariates and len(covariates) > 0:
                try:
                    # Build formula with proper column name handling
                    clean_covariates = [c for c in covariates if c in clean_data.columns]
                    if clean_covariates:
                        # Rename columns to avoid issues with special characters
                        renamed_data = clean_data.copy()
                        col_mapping = {
                            outcome_var: 'outcome',
                            group_var: 'group'
                        }
                        for i, cov in enumerate(clean_covariates):
                            col_mapping[cov] = f'cov_{i}'
                        
                        renamed_data = renamed_data.rename(columns=col_mapping)
                        
                        # Build formula
                        covariate_terms = " + ".join([f'cov_{i}' for i in range(len(clean_covariates))])
                        formula = f"outcome ~ group + {covariate_terms}"
                        
                        model = ols(formula, data=renamed_data).fit()
                        
                        results['ancova'] = {
                            'treatment_effect': model.params.get('group', None),
                            'treatment_pvalue': model.pvalues.get('group', None),
                            'r_squared': model.rsquared,
                            'adjusted_r_squared': model.rsquared_adj,
                            'formula': formula.replace('outcome', outcome_var).replace('group', group_var)
                        }
                        
                        # Replace generic covariate names with actual names in results
                        for i, cov in enumerate(clean_covariates):
                            results['ancova']['formula'] = results['ancova']['formula'].replace(f'cov_{i}', cov)
                        
                except Exception as e:
                    results['ancova_error'] = str(e)
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}

# Rest of your main() function remains the same...
def main():
    # Header
    st.markdown("""
    <div class="main-header">🔬 Interactive OpenClinical</div>
    <div class="interactive-badge">
        🎛️ Real-time Data Exploration • Dynamic Analysis • JMP/STATA-level Control
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎛️ Interactive Filters", 
            "📊 Data Explorer", 
            "📈 Visualizations", 
            "🔬 Statistical Analysis",
            "📥 Export & Reports"
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
        
        # ... (rest of your tabs remain the same) ...
        
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
            - Add/remove covariates on-the-fly
            
            **📈 Live Visualizations:**
            - Plots update with filtered data
            - Multiple plot types available
            - Interactive exploration
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 **Professional Capabilities:**
            
            **🔬 Statistical Analysis:**
            - Independent t-tests
            - ANCOVA with covariates
            - Effect size calculations
            - Confidence intervals
            
            **📥 Export Options:**
            - Filtered datasets
            - Analysis reports
            - Publication-ready results
            
            **🎯 JMP/STATA-level Control:**
            - Interactive data exploration
            - Real-time hypothesis testing
            - Professional statistical output
            """)
        
        # Example data format
        st.markdown("### 📋 **Expected Data Format:**")
        
        example_data = {
            'id': [1, 2, 3, 4, 5],
            'age': [45, 52, 38, 61, 29],
            'sex': ['F', 'M', 'F', 'M', 'F'],
            'group': [0, 1, 0, 1, 0],
            'pk1_severity': [8.5, 7.2, 9.1, 6.8, 8.0],
            'pk2_severity': [6.2, 4.1, 8.9, 3.5, 7.2],
            'response': [1, 1, 0, 1, 0]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **With this data, you could interactively:**
        - Filter by age groups (e.g., >50 vs ≤50)
        - Compare treatment groups dynamically
        - Analyze outcomes with/without baseline adjustment
        - Export filtered subsets for further analysis
        """)

if __name__ == "__main__":
    main()
