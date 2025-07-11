import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pickle
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_and_prepare_data(selected_year=2):
    data_paths = [
        'data_cleaned.csv',  # Current directory
        '/Users/rtv-lpt-129/Desktop/quantities/data_cleaned.csv',  # Absolute path
        'datasets/cleaned/data_cleaned.csv', 
        '/Users/rtv-lpt-129/Desktop/quantities/data_cleaned.csv',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '/data_cleaned.csv') 
    ]
    
    # Try to load the data from any of the paths
    data = None
    for path in data_paths:
        try:
            if os.path.exists(path):
                data = pd.read_csv(path)
                #st.success(f"Successfully loaded data from: {path}")
                break
        except Exception as e:
            continue
    
    if data is None:
        st.error("Could not load the data from any location. Please check the data file exists.")
        st.error(f"Tried these locations: {', '.join(data_paths)}")
        st.stop()
    
    data = data.copy(deep=True)
    # Filter data based on selected year
    data = data[(data.year == selected_year)].copy(deep=True)
    
    def categorize_status(row):
        daily_income = row['HH Income + Production (USD)'] / 365
        year = row['year']
        
        # Different risk thresholds by year
        if year == 1:
            threshold = 2.15  # $2.15/day for year 1
        elif year == 2:
            threshold = 3.00  # $3.00/day for year 2
        else:
            threshold = 2.15  # Default to year 1 threshold
            
        if daily_income >= threshold:
            return 1  # Above risk line
        else:
            return 0  # Below risk line
    
    data['progress_status'] = data.apply(categorize_status, axis=1)
    
    # VSLA participation
    data['vsla_participation'] = (
        (data['save_mode_rtv_vsla_cash_round'] == 1) |
        (data['save_mode_vsla_cash_round'] == 1)
    ).astype(int)
    
    # Education consolidation
    def consolidateEducation(df):
        df = df.copy(deep=True)
        dummy_columns = ['hhh_educ_level_none', 'educ_level_primary', 'educ_level_secondary', 'educ_level_tertiary']
        df[dummy_columns] = df[dummy_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        education_mapping = {
            'hhh_educ_level_none': 'None',
            'educ_level_primary': 'Primary',
            'educ_level_secondary': 'Secondary',
            'educ_level_tertiary': 'Tertiary'
        }
        
        df['education_level'] = df[dummy_columns].idxmax(axis=1).map(education_mapping)
        df = df.drop(columns=dummy_columns)
        return df
    
    data = consolidateEducation(data)
    
    from sklearn.preprocessing import LabelEncoder
    educ_label_encoder = LabelEncoder()
    data['education_level_encoded'] = educ_label_encoder.fit_transform(data['education_level'])
    data = data.drop(columns=['education_level'])
    
    # Business participation
    data['business_number'] = data['business_number'].replace('`', 0)
    data['business_number'] = data['business_number'].fillna('0')
    data['business_participation'] = data['business_number'].apply(lambda x: 1 if int(x) > 0 else 0)
    data = data.drop(columns=['business_number'])
    
    # Crop consolidation
    crops = ['sorghum', 'irish_potatoes', 'cassava', 'ground_nuts', 'maize', 'beans', 'sweet_potatoes']
    
    for crop in crops:
        season_1_col = f'Season_1_{crop}'
        season_2_col = f'Season_2_{crop}'
        consolidated_col = crop
        if season_1_col in data.columns and season_2_col in data.columns:
            data[consolidated_col] = data[[season_1_col, season_2_col]].max(axis=1)
            data = data.drop(columns=[season_1_col, season_2_col])
    
    # Handle the specific maize data issue from your notebook
    if 'sn_1_Maize_planted' in data.columns:
        mask = data['sn_1_Maize_planted'] == 'sn_2_maize_Total_Yield'
        data.loc[mask, 'sn_1_Maize_planted'] = 0
        data['sn_1_Maize_planted'] = pd.to_numeric(data['sn_1_Maize_planted'], errors='coerce')
    
    # Fill missing values for planting columns
    planting_cols = [
        'sn_1_Ground_Nuts_planted', 'Food_Banana_Qty_Planted', 'sn_1_Maize_planted',
        'sn_1_Sweet_Potatoes_planted', 'sn_1_Cassava_planted', 'sn_1_beans_planted',
        'sn_2_Ground_Nuts_planted', 'sn_2_Maize_planted', 'sn_2_Sweet_Potatoes_planted',
        'sn_2_Cassava_planted', 'sn_2_beans_planted', 'mature_coffee_planted_nasaland',
        'young_coffee_planted_nasaland', 'mature_coffee_planted_robusta',
        'young_coffee_planted_robusta'
    ]
    
    for col in planting_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    # Select relevant columns (as in your notebook)
    selected_columns = [
        'Land_size_for_Crop_Agriculture_Acres', 'farm_implements_owned',
        'tot_hhmembers', 'Distance_travelled_one_way_OPD_treatment',
        'Average_Water_Consumed_Per_Day', 'hh_water_collection_Minutes',
        'composts_num', 'vsla_participation', 'ground_nuts',
        'perennial_crops_grown_food_banana', 'sweet_potatoes',
        'perennial_crops_grown_coffee', 'irish_potatoes',
        'business_participation', 'cassava', 'hh_produce_lq_manure', 'maize',
        'sorghum', 'non_bio_waste_mgt_present', 'soap_ash_present',
        'education_level_encoded', 'tippy_tap_present', 'hhh_sex',
        'sn_1_Ground_Nuts_planted', 'Food_Banana_Qty_Planted',
        'sn_1_Maize_planted', 'sn_1_Sweet_Potatoes_planted',
        'sn_1_Cassava_planted', 'sn_1_beans_planted',
        'sn_2_Ground_Nuts_planted', 'sn_2_Maize_planted',
        'sn_2_Sweet_Potatoes_planted', 'sn_2_Cassava_planted',
        'sn_2_beans_planted', 'mature_coffee_planted_nasaland',
        'young_coffee_planted_nasaland', 'mature_coffee_planted_robusta',
        'young_coffee_planted_robusta', 'progress_status'
    ]
    
    # Filter columns that exist in the data
    available_columns = [col for col in selected_columns if col in data.columns]
    data = data[available_columns]
    
    # Drop additional columns if they exist
    columns_to_drop = ['pre_district', 'region_Mid_West', 'region_South_West', 'region_Eastern', 'year']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    return data

@st.cache_resource
def load_model(selected_year=2):
    """Load the pre-trained RandomForest model from pickle file based on selected year"""
    
    # Define possible model paths based on year
    model_paths = [
        f'random_forest_model_{selected_year}.pkl',  # Current directory
        f'/Users/rtv-lpt-129/Desktop/quantities/random_forest_model_{selected_year}.pkl',  # Absolute path
        f'/Users/rtv-lpt-129/Desktop/quantities/random_forest_model_{selected_year}.pkl', 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f'random_forest_model_{selected_year}.pkl')
    ]
    
    scaler_paths = [
        f'scaler_{selected_year}.pkl',  # Current directory
        f'/Users/rtv-lpt-129/Desktop/quantities/scaler_{selected_year}.pkl',  # Absolute path
        f'/Users/rtv-lpt-129/Desktop/quantities/scaler_{selected_year}.pkl', 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f'scaler_{selected_year}.pkl')
    ]
    
    fallback_scaler_paths = [
        'scaler.pkl',  # Current directory
        '/Users/rtv-lpt-129/Desktop/quantities/scaler.pkl',  # Absolute path
        '/Users/rtv-lpt-129/Desktop/Data_Preprocessing/workmate-data/scaler.pkl', 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')
    ]
    
    # Try to load the model from any of the paths
    model = None
    model_loaded_from = None
    for path in model_paths:
        try:
            if os.path.exists(path):
                try:
                    import joblib
                    model = joblib.load(path)
                    #st.success(f"Successfully loaded Year {selected_year} model from: {path}")
                    pass
                    model_loaded_from = path
                    break
                except:
                    # Fallback to pickle if joblib fails
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    #st.success(f"Successfully loaded Year {selected_year} model from: {path}")
                    model_loaded_from = path
                    break
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            continue
        except Exception as e:
            continue
    
    scaler = None
    scaler_loaded_from = None
    all_scaler_paths = scaler_paths + fallback_scaler_paths
    
    for path in all_scaler_paths:
        try:
            if os.path.exists(path):
                # Try pickle first (this is more common for scalers)
                try:
                    with open(path, 'rb') as f:
                        scaler = pickle.load(f)
                    if 'scaler_' in path:
                        #st.success(f"Successfully loaded Year {selected_year} scaler from: {path}")
                        pass
                    else:
                        #st.info(f"Using generic scaler from: {path}")
                        pass
                    scaler_loaded_from = path
                    break
                except:
                    # Fallback to joblib if pickle fails
                    import joblib
                    scaler = joblib.load(path)
                    if 'scaler_' in path:
                        #st.success(f" Successfully loaded Year {selected_year} scaler from: {path}")
                        pass
                    else:
                        #st.info(f"Using generic scaler from: {path}")
                        pass
                    scaler_loaded_from = path
                    break
        except Exception as e:
            continue
        
    return model, scaler

def get_feature_names(data):
    """Get feature names from the dataset"""
    # Drop the target variable to get feature names
    X = data.drop(columns=['progress_status', 'HH Income + Production (USD)'] 
                  if 'HH Income + Production (USD)' in data.columns 
                  else ['progress_status'])
    return X.columns.tolist()

def create_prediction_surface(model, scaler, feature_names, land_size_range, crop_qty_range, 
                            crop_feature, baseline_features):
    """Create a prediction surface for land size vs crop quantity"""
    
    # Create meshgrid
    land_sizes = np.linspace(land_size_range[0], land_size_range[1], 40)
    crop_qtys = np.linspace(crop_qty_range[0], crop_qty_range[1], 40)
    
    Land_mesh, Crop_mesh = np.meshgrid(land_sizes, crop_qtys)
    
    # Prepare prediction data
    predictions = np.zeros_like(Land_mesh)
    
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    
    # If we have model features, ensure our feature vector matches
    if model_features is not None and len(model_features) != len(feature_names):
        st.warning(f"Feature mismatch: Model expects {len(model_features)} features, but data has {len(feature_names)}.")
        
        # Create a mapping from data features to model features
        feature_mapping = {}
        for i, feature in enumerate(feature_names):
            if feature in model_features:
                feature_mapping[i] = np.where(model_features == feature)[0][0]
        
        # Create a function to map our features to model features
        def map_features(features):
            model_input = np.zeros(len(model_features))
            for data_idx, model_idx in feature_mapping.items():
                if data_idx < len(features):
                    model_input[model_idx] = features[data_idx]
            return model_input
    else:
        map_features = lambda x: x
    
    for i in range(Land_mesh.shape[0]):
        for j in range(Land_mesh.shape[1]):
            # Create feature vector
            features = baseline_features.copy()
            
            # Find indices for land size and crop quantity
            land_idx = feature_names.index('Land_size_for_Crop_Agriculture_Acres') if 'Land_size_for_Crop_Agriculture_Acres' in feature_names else -1
            crop_idx = feature_names.index(crop_feature) if crop_feature in feature_names else -1
            
            if land_idx >= 0 and land_idx < len(features):
                features[land_idx] = Land_mesh[i, j]
            if crop_idx >= 0 and crop_idx < len(features):
                features[crop_idx] = Crop_mesh[i, j]
            
            # Map features to model's expected format
            features_mapped = map_features(features)
            
            try:
                # Scale and predict
                features_scaled = scaler.transform([features_mapped])
                prob = model.predict_proba(features_scaled)[0][1]
                predictions[i, j] = prob
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return None, None, None
    
    return Land_mesh, Crop_mesh, predictions, map_features

def find_risk_line_boundary(Land_mesh, Crop_mesh, predictions, threshold=0.5):
    """Find the boundary where households transition from below to above risk line"""
    boundary_points = []
    
    for i in range(predictions.shape[0] - 1):
        for j in range(predictions.shape[1] - 1):
            # Check if we cross the threshold
            current = predictions[i, j]
            right = predictions[i, j + 1]
            down = predictions[i + 1, j]
            
            if (current < threshold < right) or (current > threshold > right):
                boundary_points.append((Land_mesh[i, j], Crop_mesh[i, j]))
            if (current < threshold < down) or (current > threshold > down):
                boundary_points.append((Land_mesh[i, j], Crop_mesh[i, j]))
    
    return boundary_points


def create_one_way_pdp(model, scaler, feature_names, feature_name, baseline_features, feature_range, n_points=50):
    """Create one-way partial dependence plot for a single feature"""
    
    # Create feature values for PDP
    feature_values = np.linspace(feature_range[0], feature_range[1], n_points)
    
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    
    if model_features is not None and len(model_features) != len(feature_names):
        feature_mapping = {}
        for i, feature in enumerate(feature_names):
            if feature in model_features:
                feature_mapping[i] = np.where(model_features == feature)[0][0]
        
        def map_features(features):
            model_input = np.zeros(len(model_features))
            for data_idx, model_idx in feature_mapping.items():
                if data_idx < len(features):
                    model_input[model_idx] = features[data_idx]
            return model_input
    else:
        
        map_features = lambda x: x
    
    predictions = []
    
    for feature_val in feature_values:
        # Create feature vector
        features = baseline_features.copy()
        
        # Find index for the feature
        feature_idx = feature_names.index(feature_name) if feature_name in feature_names else -1
        
        # Update feature if index is valid
        if feature_idx >= 0 and feature_idx < len(features):
            features[feature_idx] = feature_val
        
        features_mapped = map_features(features)
        
        try:
            # Scale and predict
            features_scaled = scaler.transform([features_mapped])
            prob = model.predict_proba(features_scaled)[0][1]
            predictions.append(prob)
        except Exception as e:
            st.error(f"PDP Prediction error: {str(e)}")
            return None, None
    
    return feature_values, predictions

def create_simple_combined_plot(model, scaler, feature_names, selected_crop, baseline_features, 
                              land_size_range, crop_qty_range, n_points=30):
    """Create a simple plot with crop quantity on x-axis and land size on y-axis"""
    
    # Create meshgrid for crop quantity (x) and land size (y)
    crop_values = np.linspace(crop_qty_range[0], crop_qty_range[1], n_points)
    land_values = np.linspace(land_size_range[0], land_size_range[1], n_points)
    
    # Create meshgrid
    Crop_mesh, Land_mesh = np.meshgrid(crop_values, land_values)
    
    # Check if model has feature_names_in_ attribute
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    
    # Create feature mapping function
    if model_features is not None and len(model_features) != len(feature_names):
        feature_mapping = {}
        for i, feature in enumerate(feature_names):
            if feature in model_features:
                feature_mapping[i] = np.where(model_features == feature)[0][0]
        
        def map_features(features):
            model_input = np.zeros(len(model_features))
            for data_idx, model_idx in feature_mapping.items():
                if data_idx < len(features):
                    model_input[model_idx] = features[data_idx]
            return model_input
    else:
        map_features = lambda x: x
    
    # Calculate predictions for each combination
    predictions = np.zeros_like(Crop_mesh)
    
    for i in range(Crop_mesh.shape[0]):
        for j in range(Crop_mesh.shape[1]):
            # Create feature vector
            features = baseline_features.copy()
            
            # Find indices for land size and crop quantity
            land_idx = feature_names.index('Land_size_for_Crop_Agriculture_Acres') if 'Land_size_for_Crop_Agriculture_Acres' in feature_names else -1
            crop_idx = feature_names.index(selected_crop) if selected_crop in feature_names else -1
            
            # Update features if indices are valid
            if land_idx >= 0 and land_idx < len(features):
                features[land_idx] = Land_mesh[i, j]
            if crop_idx >= 0 and crop_idx < len(features):
                features[crop_idx] = Crop_mesh[i, j]
            
            # Map features to model's expected format
            features_mapped = map_features(features)
            
            try:
                # Scale and predict
                features_scaled = scaler.transform([features_mapped])
                prob = model.predict_proba(features_scaled)[0][1]  # Probability of being above risk line
                predictions[i, j] = prob
            except Exception as e:
                predictions[i, j] = 0  # Default to 0 if prediction fails
    
    # Create the plot
    fig = go.Figure()
    
    # Add scatter plot with color representing probability
    fig.add_trace(go.Scatter(
        x=Crop_mesh.flatten(),
        y=Land_mesh.flatten(),
        mode='markers',
        marker=dict(
            size=8,
            color=predictions.flatten(),
            colorscale='RdYlGn',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Probability of<br>Escaping Risk", tickformat='.0%'),
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        hovertemplate='<b>Risk Escape Probability</b><br>' +
                    f'Crop Quantity: %{{x:.0f}} {get_crop_unit(selected_crop)}<br>' +
                    'Land Size: %{y:.1f} acres<br>' +
                    'Escape Probability: %{marker.color:.1%}<extra></extra>',
        name=''
    ))
    
    fig.update_layout(
        title=f"Risk Escape Probability: {format_crop_name(selected_crop)} vs Land Size",
        xaxis_title=f"Crop Quantity ({get_crop_unit(selected_crop)})",
        yaxis_title="Land Size (Acres)",
        height=600,
        showlegend=False
    )
    
    return fig, predictions, Crop_mesh, Land_mesh

def create_line_plot(model, scaler, feature_names, selected_crop, baseline_features, 
                   land_size_range, crop_qty_range, selected_land_sizes, n_points=50):
    """Create a line plot with crop quantity on x-axis and different lines for different land sizes"""
    
    # Create crop quantity values for x-axis
    crop_values = np.linspace(crop_qty_range[0], crop_qty_range[1], n_points)
    
    # Use the selected land sizes instead of predefined ones
    land_sizes = selected_land_sizes
    
    # Check if model has feature_names_in_ attribute
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    
    # Create feature mapping function
    if model_features is not None and len(model_features) != len(feature_names):
        feature_mapping = {}
        for i, feature in enumerate(feature_names):
            if feature in model_features:
                feature_mapping[i] = np.where(model_features == feature)[0][0]
        
        def map_features(features):
            model_input = np.zeros(len(model_features))
            for data_idx, model_idx in feature_mapping.items():
                if data_idx < len(features):
                    model_input[model_idx] = features[data_idx]
            return model_input
    else:
        map_features = lambda x: x
    
    # Create the plot
    fig = go.Figure()
    
    # Colors for different land sizes - expand color palette for more options
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    # Store all predictions for analysis
    all_predictions = []
    all_crop_values = []
    all_land_values = []
    
    # Create a line for each selected land size
    for i, land_size in enumerate(land_sizes):
        predictions_for_land = []
        
        for crop_qty in crop_values:
            # Create feature vector
            features = baseline_features.copy()
            
            # Find indices for land size and crop quantity
            land_idx = feature_names.index('Land_size_for_Crop_Agriculture_Acres') if 'Land_size_for_Crop_Agriculture_Acres' in feature_names else -1
            crop_idx = feature_names.index(selected_crop) if selected_crop in feature_names else -1
            
            # Update features if indices are valid
            if land_idx >= 0 and land_idx < len(features):
                features[land_idx] = land_size
            if crop_idx >= 0 and crop_idx < len(features):
                features[crop_idx] = crop_qty
            
            # Map features to model's expected format
            features_mapped = map_features(features)
            
            try:
                # Scale and predict
                features_scaled = scaler.transform([features_mapped])
                prob = model.predict_proba(features_scaled)[0][1]  # Probability of being above risk line
                predictions_for_land.append(prob)
                
                # Store for overall analysis
                all_predictions.append(prob)
                all_crop_values.append(crop_qty)
                all_land_values.append(land_size)
            except Exception as e:
                predictions_for_land.append(0)  # Default to 0 if prediction fails
        
        # Add line for this land size
        fig.add_trace(go.Scatter(
            x=crop_values,
            y=predictions_for_land,
            mode='lines+markers',
            name=f'{land_size:.1f} acres',
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=6, color=colors[i % len(colors)]),
            hovertemplate=f'<b>Land Size: {land_size:.1f} acres</b><br>' +
                        f'Crop Quantity: %{{x:.0f}} {get_crop_unit(selected_crop)}<br>' +
                        'Escape Probability: %{y:.1%}<extra></extra>'
        ))
    
    # Add horizontal line at 50% threshold
    fig.add_hline(
        y=0.5, 
        line_dash="dot", 
        line_color="red",
        annotation_text="50% Risk Escape Threshold",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Risk Escape Probability by {format_crop_name(selected_crop)} Quantity",
        xaxis_title=f"Crop Quantity ({get_crop_unit(selected_crop)})",
        yaxis_title="Probability of Escaping Risk",
        height=600,
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=True,
        legend=dict(
            title="Land Size",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig, np.array(all_predictions), np.array(all_crop_values), np.array(all_land_values)

def create_single_line_plot(model, scaler, feature_names, selected_crop, baseline_features,land_size_range, crop_qty_range, selected_land_size, n_points=50):
    """Create a single line plot with crop quantity on x-axis for one specific land size"""
    
    # Create crop quantity values for x-axis
    crop_values = np.linspace(crop_qty_range[0], crop_qty_range[1], n_points)
    
    # Check if model has feature_names_in_ attribute
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    
    # Create feature mapping function
    if model_features is not None and len(model_features) != len(feature_names):
        feature_mapping = {}
        for i, feature in enumerate(feature_names):
            if feature in model_features:
                feature_mapping[i] = np.where(model_features == feature)[0][0]
        
        def map_features(features):
            model_input = np.zeros(len(model_features))
            for data_idx, model_idx in feature_mapping.items():
                if data_idx < len(features):
                    model_input[model_idx] = features[data_idx]
            return model_input
    else:
        map_features = lambda x: x
    
    # Create predictions for the selected land size
    predictions_for_land = []
    all_crop_values = []
    all_land_values = []
    
    for crop_qty in crop_values:
        # Create feature vector
        features = baseline_features.copy()
        
        # Find indices for land size and crop quantity
        land_idx = feature_names.index('Land_size_for_Crop_Agriculture_Acres') if 'Land_size_for_Crop_Agriculture_Acres' in feature_names else -1
        crop_idx = feature_names.index(selected_crop) if selected_crop in feature_names else -1
        
        # Update features if indices are valid
        if land_idx >= 0 and land_idx < len(features):
            features[land_idx] = selected_land_size
        if crop_idx >= 0 and crop_idx < len(features):
            features[crop_idx] = crop_qty
        
        # Map features to model's expected format
        features_mapped = map_features(features)
        
        try:
            # Scale and predict
            features_scaled = scaler.transform([features_mapped])
            prob = model.predict_proba(features_scaled)[0][1]  # Probability of being above risk line
            predictions_for_land.append(prob)
            
            # Store for analysis
            all_crop_values.append(crop_qty)
            all_land_values.append(selected_land_size)
        except Exception as e:
            predictions_for_land.append(0)  # Default to 0 if prediction fails
            all_crop_values.append(crop_qty)
            all_land_values.append(selected_land_size)
    
    # Create the plot
    fig = go.Figure()
    
    # Add single line for the selected land size
    fig.add_trace(go.Scatter(
        x=crop_values,
        y=predictions_for_land,
        mode='lines+markers',
        name=f'{selected_land_size:.1f} acres',
        line=dict(color='#4ECDC4', width=4),
        marker=dict(size=8, color='#4ECDC4'),
        hovertemplate=f'<b>Land Size: {selected_land_size:.1f} acres</b><br>' +
                    f'Crop Quantity: %{{x:.0f}} {get_crop_unit(selected_crop)}<br>' +
                    'Risk Escape Probability: %{y:.1%}<extra></extra>'
    ))
    
    # Add horizontal line at 50% threshold
    fig.add_hline(
        y=0.5, 
        line_dash="dot", 
        line_color="red",
        annotation_text="50% Risk Escape Threshold",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Risk Escape Probability by {format_crop_name(selected_crop)} Quantity<br><sub>Land Size: {selected_land_size:.1f} acres</sub>",
        xaxis_title=f"Crop Quantity ({get_crop_unit(selected_crop)})",
        yaxis_title="Probability of Escaping Risk",
        height=600,
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=False
    )
    
    return fig, np.array(predictions_for_land), np.array(all_crop_values), np.array(all_land_values)

def create_general_relationship_plot(data, selected_crop, current_year):
    """Create a general plot showing the relationship between crop quantity and land size"""
    
    # Filter data for the selected crop and land size
    if selected_crop in data.columns and 'Land_size_for_Crop_Agriculture_Acres' in data.columns:
        # Create a subset with non-zero crop quantities for better visualization
        plot_data = data[
            (data[selected_crop] > 0) & 
            (data['Land_size_for_Crop_Agriculture_Acres'] > 0)
        ].copy()
        
        if len(plot_data) == 0:
            return None
        
        # Add poverty status for color coding
        plot_data['poverty_status'] = plot_data['progress_status'].map({
            0: 'Below Risk Line',
            1: 'Above Risk Line'
        })
        
        return plot_data
    
    return None

def format_crop_name(crop_name):
    if crop_name == 'Food_Banana_Qty_Planted':
        return 'Food Banana Quantity'
    else:
        season = "Season 1" if 'sn_1_' in crop_name else "Season 2"
        crop = crop_name.replace('sn_1_', '').replace('sn_2_', '').replace('_planted', '').replace('_', ' ').title()
        return f"{season} {crop}"

def get_crop_unit(crop_name):
    """Get the appropriate unit of measure for each crop"""
    if 'Cassava' in crop_name or 'cassava' in crop_name:
        return 'stems'
    elif 'Banana' in crop_name or 'banana' in crop_name:
        return 'suckers'
    elif 'Sweet_Potatoes' in crop_name or 'sweet_potatoes' in crop_name or 'Sweet Potatoes' in crop_name:
        return 'bags'
    else:
        return 'kg'

def main():
    st.set_page_config(page_title="Crop Risk Analysis Dashboard", layout="wide")
    
    st.title("Crop Quantity & Land Size Relationship Analysis")
    
    # Year selection at the top
    st.sidebar.header("Analysis Settings")
    selected_year = st.sidebar.selectbox(
        "Select Analysis Year:",
        options=[1, 2],
        index=1,  # Default to year 2
        help="Choose which year's data and model to use for analysis"
    )
    
    # Determine current year and threshold for display
    current_year = selected_year
    threshold = 2.15 if current_year == 1 else 3.00
    
    # Add year indicator
    st.info(f"**Currently analyzing Year {current_year}** with risk threshold of **${threshold}/day**")
    
    st.markdown(f"**Year {current_year} Analysis - Explore which combinations of land size and crop quantities help households escape risk (${threshold}/day threshold)**")
    
    # Load data and pre-trained model
    with st.spinner(f"Loading Year {current_year} data and model..."):
        data = load_and_prepare_data(selected_year=current_year)
        
        model, scaler = load_model(selected_year=current_year)
        
        # Get feature names from the data
        feature_names = get_feature_names(data)
        
        # If the scaler is not pre-trained, fit it with the data
        if not hasattr(scaler, 'mean_') or len(scaler.mean_) == 0:
            X = data.drop(columns=['progress_status', 'HH Income + Production (USD)'] 
                         if 'HH Income + Production (USD)' in data.columns 
                         else ['progress_status'])
            scaler.fit(X)
        else:
            pass
    
    st.sidebar.write("**Target Categories:**")
    st.sidebar.write(f"• 0 = Below risk line (<${threshold}/day)")
    st.sidebar.write(f"• 1 = Above risk line (≥${threshold}/day)")
    
    # Crop selection
    st.sidebar.header("Analysis Parameters")
    
    # Available crop features from your model
    crop_features = [
        'sn_1_Ground_Nuts_planted', 'sn_1_Maize_planted', 'sn_1_Sweet_Potatoes_planted',
        'sn_1_Cassava_planted', 'sn_1_beans_planted', 'sn_2_Ground_Nuts_planted',
        'sn_2_Maize_planted', 'sn_2_Sweet_Potatoes_planted', 'sn_2_Cassava_planted',
        'sn_2_beans_planted', 'Food_Banana_Qty_Planted'
    ]
    
    # Filter available crop features
    available_crop_features = [f for f in crop_features if f in feature_names]
    
    # Create better display names
    def format_crop_name(crop_name):
        if crop_name == 'Food_Banana_Qty_Planted':
            return 'Food Banana Quantity'
        else:
            season = "Season 1" if 'sn_1_' in crop_name else "Season 2"
            crop = crop_name.replace('sn_1_', '').replace('sn_2_', '').replace('_planted', '').replace('_', ' ').title()
            return f"{season} {crop}"
    
    def get_crop_unit(crop_name):
        """Get the appropriate unit of measure for each crop"""
        if 'Cassava' in crop_name or 'cassava' in crop_name:
            return 'stems'
        elif 'Banana' in crop_name or 'banana' in crop_name:
            return 'suckers'
        elif 'Sweet_Potatoes' in crop_name or 'sweet_potatoes' in crop_name or 'Sweet Potatoes' in crop_name:
            return 'bags'
        else:
            return 'kg'
    
    selected_crop = st.sidebar.selectbox(
        "Select Crop to Analyze:",
        available_crop_features,
        format_func=format_crop_name
    )
    
    # Range controls
    land_size_max = data['Land_size_for_Crop_Agriculture_Acres'].max()
    crop_qty_max = data[selected_crop].max() if selected_crop in data.columns else 100
    
    land_size_range = st.sidebar.slider(
        "Land Size Range (Acres):",
        0.0, float(land_size_max), (0.1, min(10.0, float(land_size_max))), 0.1
    )
    
    crop_qty_range = st.sidebar.slider(
        f"Crop Quantity Range:",
        0.0, float(crop_qty_max), (5.0, min(50.0, float(crop_qty_max))), 1.0
    )
    
    # Single land size selection for the plot
    st.sidebar.subheader("Land Size Selection")
    selected_land_size = st.sidebar.slider(
        "Select Land Size (acres):",
        min_value=land_size_range[0],
        max_value=land_size_range[1],
        value=(land_size_range[0] + land_size_range[1]) / 2,  
        step=0.1,
        help="Choose the land size to analyze - the plot will show how risk escape probability changes with crop quantity for this specific land size"
    )
    
    # Baseline household characteristics
    st.sidebar.subheader("Household Characteristics")
    
    baseline_features = []
    for feature in feature_names:
        if feature == 'Land_size_for_Crop_Agriculture_Acres' or feature == selected_crop:
            baseline_features.append(0)
        elif feature in data.columns:
            median_val = data[feature].median()
            baseline_features.append(median_val)
        else:
            baseline_features.append(0)
    
    # Allow user to modify key baseline features
    if 'tot_hhmembers' in feature_names:
        household_size = st.sidebar.slider("Household Size:", 1, 15, int(data['tot_hhmembers'].median()))
        baseline_features[feature_names.index('tot_hhmembers')] = household_size
    
    if 'vsla_participation' in feature_names:
        vsla_participation = st.sidebar.selectbox("VSLA Participation:", [0, 1], index=1)
        baseline_features[feature_names.index('vsla_participation')] = vsla_participation
    
    if 'business_participation' in feature_names:
        business_participation = st.sidebar.selectbox("Business Participation:", [0, 1], index=0)
        baseline_features[feature_names.index('business_participation')] = business_participation
    
    # Create prediction surface for the heatmap
    with st.spinner("Generating predictions..."):
        result = create_prediction_surface(
            model, scaler, feature_names, land_size_range, crop_qty_range, 
            selected_crop, baseline_features
        )
        
        # Check if prediction surface creation failed
        if result is None or len(result) < 3 or any(x is None for x in result[:3]):
            st.error("Failed to generate predictions. There might be an issue with the model or data compatibility.")
            st.stop()
        
        # Unpack results
        if len(result) == 4:
            Land_mesh, Crop_mesh, predictions, map_features_func = result
        else:
            Land_mesh, Crop_mesh, predictions = result[:3]
            map_features_func = lambda x: x 
    
    # Find risk line boundary
    boundary_points = find_risk_line_boundary(Land_mesh, Crop_mesh, predictions, threshold=0.5)

    # Create main visualization - Original Heatmap
    st.subheader("Overall Risk Escape Probability Heatmap")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 2D Heatmap with risk line
        fig_2d = go.Figure()
        
        # Add heatmap
        fig_2d.add_trace(go.Heatmap(
            z=predictions,
            x=Land_mesh[0, :],
            y=Crop_mesh[:, 0],
            colorscale='RdYlGn',
            colorbar=dict(title="Probability of Escaping Risk"),
            name="Probability Heatmap"
        ))
        
        # Add risk line boundary
        if boundary_points:
            boundary_x, boundary_y = zip(*boundary_points)
            fig_2d.add_trace(go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode='markers',
                marker=dict(size=4, color='red'),
                name='Risk Line (50% threshold)'
            ))
        
        fig_2d.update_layout(
            title=f"Year {current_year} Risk Escape Probability: Land Size vs {format_crop_name(selected_crop)}",
            xaxis_title="Land Size (Acres)",
            yaxis_title=f"Crop Quantity ({format_crop_name(selected_crop)}) - {get_crop_unit(selected_crop)}",
            height=600
        )
        
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # Risk escape zones
        high_escape = (predictions >= 0.7).sum() / predictions.size
        medium_escape = ((predictions >= 0.5) & (predictions < 0.7)).sum() / predictions.size
        low_escape = (predictions < 0.5).sum() / predictions.size
        
        st.markdown(f"""
        <div style="text-align: center;">
        <h2>Risk Escape Zones</h2>
        🟢 <strong>High Escape (≥70%):</strong> {high_escape:.1%} of combinations<br>
        🟡 <strong>Medium Escape (50-70%):</strong> {medium_escape:.1%} of combinations<br>
        🔴 <strong>Low Escape (<50%):</strong> {low_escape:.1%} of combinations
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Heatmap Insights")
        
        # Find optimal combinations from heatmap
        max_prob_idx = np.unravel_index(np.argmax(predictions), predictions.shape)
        optimal_land = Land_mesh[max_prob_idx]
        optimal_crop = Crop_mesh[max_prob_idx]
        max_prob = predictions[max_prob_idx]
        
        st.metric(
            "Best Risk Escape Scenario",
            f"{max_prob:.1%}",
            f"Land: {optimal_land:.1f} acres, Crop: {optimal_crop:.0f}"
        )
        
        # Find minimum viable combinations (50% threshold)
        viable_mask = predictions >= 0.5
        if np.any(viable_mask):
            viable_land = Land_mesh[viable_mask]
            viable_crop = Crop_mesh[viable_mask]
            min_land_viable = viable_land.min()
            min_crop_at_min_land = viable_crop[viable_land == min_land_viable].min()
            
            st.metric(
                "Minimum Viable Scenario",
                "≥50% Risk escape probability",
                f"Land: {min_land_viable:.1f} acres, Crop: {min_crop_at_min_land:.0f}"
            )
        
        # Statistics
        st.subheader("Heatmap Statistics")
        st.write(f"**Average Risk Escape Probability:** {predictions.mean():.1%}")
        st.write(f"**Minimum Probability:** {predictions.min():.1%}")
        st.write(f"**Maximum Probability:** {predictions.max():.1%}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_dict = {feature: importance for feature, importance in 
                              zip(model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feature_names, 
                                 model.feature_importances_)}
            # Create dataframe for plotting
            importance_df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            }).sort_values('Importance', ascending=False).head(5)
            
            # st.subheader("Top Feature Importance")
            fig_importance = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f"Year {current_year} Feature Importance"
            )
            fig_importance.update_layout(height=250)
            st.plotly_chart(fig_importance, use_container_width=True)

    # Add separator
    st.markdown("---")

    # Create single line plot visualization
    st.subheader("Specific Land Size Analysis")
    
    # Generate the single line plot with current household characteristics
    with st.spinner("Generating single land size analysis..."):
        result = create_single_line_plot(
            model, scaler, feature_names, selected_crop, baseline_features,
            land_size_range, crop_qty_range, selected_land_size
        )
        
        if result is not None:
            fig, predictions_single, crop_values, land_values = result
            
            # Display the plot first
            st.plotly_chart(fig, use_container_width=True)
            
            # Show selected land size info
            st.info(f" **Analyzing land size:** {selected_land_size:.1f} acres with current household characteristics")
            
            # Analysis insights - moved below the plot
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Analysis Insights")
                
                # Find optimal crop quantity for this land size
                max_prob_idx = np.argmax(predictions_single)
                optimal_crop_qty = crop_values[max_prob_idx]
                max_prob = predictions_single[max_prob_idx]
                
                st.write(f"**Optimal Combination for Current Household:**")
                st.write(f"• **{optimal_crop_qty:.0f} {get_crop_unit(selected_crop)}** of {format_crop_name(selected_crop)}")
                st.write(f"• **{max_prob:.1%}** Risk escape probability")
                st.write(f"• **With {selected_land_size:.1f} acres** of land")
                
                # Find minimum viable crop quantity (50% threshold)
                viable_mask = predictions_single >= 0.5
                if np.any(viable_mask):
                    viable_crop_qtys = crop_values[viable_mask]
                    min_viable_crop = viable_crop_qtys.min()
                    
                    st.write(f"** Minimum Viable Crop Quantity (≥50% Risk escape probability):**")
                    st.write(f"• **{min_viable_crop:.0f} {get_crop_unit(selected_crop)}** of {format_crop_name(selected_crop)}")
                    st.write(f"• With current household characteristics")
                else:
                    st.write("** No crop quantity reaches 50% Risk escape probability for this household profile**")
                
                # Show impact of household characteristics
                st.write("**Household Characteristics Impact:**")
                if 'household_size' in locals():
                    if household_size > data['tot_hhmembers'].median():
                        st.write(f"• **Larger household** ({household_size} vs {data['tot_hhmembers'].median():.1f} avg) may require more resources")
                    elif household_size < data['tot_hhmembers'].median():
                        st.write(f"• **Smaller household** ({household_size} vs {data['tot_hhmembers'].median():.1f} avg) may have advantage")
                
                if vsla_participation == 1:
                    st.write("• **VSLA participation** likely improves risk escape probability")
                else:
                    st.write("• **No VSLA participation** - consider joining for better outcomes")
                
                if business_participation == 1:
                    st.write("• **Business participation** provides additional income diversification")
                else:
                    st.write("• **No business participation** - crop production is primary income source")
                

            
            with col2:
                st.subheader("Key Metrics")
                # Best scenario metric
                st.metric(
                    "Best Risk Escape Probability",
                    f"{max_prob:.1%}",
                    f"At {optimal_crop_qty:.0f} {get_crop_unit(selected_crop)}"
                )
                
                # Minimum viable scenario
                if np.any(viable_mask):
                    st.metric(
                        "Minimum Viable Quantity",
                        f"{min_viable_crop:.0f} {get_crop_unit(selected_crop)}",
                        "For 50% Risk escape probability"
                    )
                
                # Statistics
                st.write(f"**Average Risk Escape Probability:** {predictions_single.mean():.1%}")
                st.write(f"**Minimum Probability:** {predictions_single.min():.1%}")
                st.write(f"**Maximum Probability:** {predictions_single.max():.1%}")
                
                # Risk escape zones
                high_escape = (predictions_single >= 0.7).sum() / len(predictions_single)
                medium_escape = ((predictions_single >= 0.5) & (predictions_single < 0.7)).sum() / len(predictions_single)
                low_escape = (predictions_single < 0.5).sum() / len(predictions_single)
        else:
            st.error("Failed to generate the single land size analysis plot.")

    st.markdown("---")

    # One-way Partial Dependence Plot using sklearn
    st.subheader("One-Way Partial Dependence Plot")

    # Generate one-way PDP using sklearn
    with st.spinner("Generating one-way PDP"):
        try:
            from sklearn.inspection import partial_dependence
            
            # Prepare the data for PDP
            X = data.drop(columns=['progress_status'])
            y = data['progress_status']
            
            # Get the feature index for the selected crop
            if selected_crop in X.columns:
                feature_idx = X.columns.get_loc(selected_crop)
                
                # Calculate partial dependence using sklearn - this shows relationship with target
                pdp_result = partial_dependence(
                    model, 
                    X=X, 
                    features=[feature_idx],
                    grid_resolution=50
                )
                
                # Extract results - sklearn returns a Bunch object with 'average' and 'grid_values'
                if hasattr(pdp_result, 'average') and hasattr(pdp_result, 'grid_values'):
                    # Modern sklearn (1.0+)
                    pdp_values = pdp_result.average[0]
                    feature_values = pdp_result.grid_values[0]
                elif isinstance(pdp_result, tuple) and len(pdp_result) >= 2:
                    # Older sklearn versions return tuple
                    pdp_values = pdp_result[0][0]
                    feature_values = pdp_result[1][0]
                else:
                    raise ValueError("Unknown sklearn partial_dependence return format")
                
                # Convert to numpy arrays
                pdp_values = np.array(pdp_values)
                feature_values = np.array(feature_values)
                
                # Create the PDP plot - sklearn style
                fig_pdp = go.Figure()
                
                # Add the main PDP line
                fig_pdp.add_trace(go.Scatter(
                    x=feature_values,
                    y=pdp_values,
                    mode='lines',
                    name='',
                    line=dict(color='green', width=3),
                    hovertemplate=f'<b>{format_crop_name(selected_crop)}:</b> %{{x:.1f}}<br>' +
                                '<b>Partial dependence:</b> %{y:.4f}<extra></extra>'
                ))
                
                fig_pdp.update_layout(
                    title=f"Partial dependence of '{format_crop_name(selected_crop)}'",
                    xaxis_title=f"{format_crop_name(selected_crop)} ({get_crop_unit(selected_crop)})",
                    yaxis_title="Partial dependence",
                    height=400,
                    width=600,
                    showlegend=False,
                    plot_bgcolor='white',
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='lightgray',
                        linecolor='black',
                        mirror=True
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='lightgray',
                        linecolor='black',
                        mirror=True
                    ),
                    font=dict(size=12)
                )
                
                # Display the plot centered
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.plotly_chart(fig_pdp, use_container_width=False)
                
                # Add interpretation

                 
            else:
                st.error(f"Feature '{selected_crop}' not found in dataset columns")
                
        except Exception as e:
            st.error(f"Failed to generate sklearn PDP: {str(e)}")
            st.write("**Error details:**", str(e))
            
            # Simple fallback: Show actual data relationship
            if selected_crop in data.columns:
                st.write("**Fallback: Showing actual data relationship**")
                fig_fallback = px.scatter(
                    data, 
                    x=selected_crop, 
                    y='progress_status',
                    color='progress_status',
                    title=f"Actual Data: {format_crop_name(selected_crop)} vs Risk Status",
                    labels={
                        selected_crop: f"{format_crop_name(selected_crop)} ({get_crop_unit(selected_crop)})",
                        'progress_status': 'Risk Status (0=Below, 1=Above)'
                    }
                )
                st.plotly_chart(fig_fallback, use_container_width=True)


if __name__ == "__main__":
    main() 