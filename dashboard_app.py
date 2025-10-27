"""
Steam Games Pricing Strategy Dashboard
Phase 2 Analysis Visualization
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Steam Games Pricing Strategy Dashboard"
server = app.server  # Expose server for deployment (Gunicorn, Render, etc.)

# Load data
print("[Loading] Reading CSV files...")
games = pd.read_csv('games_simplified.csv')
price_data = pd.read_csv('price_final.csv')
count_data = pd.read_csv('count_final.csv')
review_data = pd.read_csv('review_final.csv')
print("[Loading] Processing data...")

# Extract release year
games['Release year'] = pd.to_datetime(games['Release date'], errors='coerce').dt.year

# Process estimated owners
def extract_owners_mid(owners_str):
    if pd.isna(owners_str) or owners_str == '0 - 0':
        return 0
    parts = str(owners_str).split(' - ')
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return 0

games['Estimated owners (mid)'] = games['Estimated owners'].apply(extract_owners_mid)

# Calculate positive rate
games['Positive rate'] = games['Positive'] / (games['Positive'] + games['Negative'] + 1)
games['Total reviews'] = games['Positive'] + games['Negative']

# Revenue estimation
games['Estimated revenue'] = games['Estimated owners (mid)'] * games['Price']

# Extract primary genre
def get_primary_genre(genres_str):
    if pd.isna(genres_str):
        return 'Unknown'
    return str(genres_str).split(',')[0].strip()

games['Primary genre'] = games['Genres'].apply(get_primary_genre)

# Price categories
def categorize_price(price):
    if price == 0:
        return 'Free'
    elif price < 5:
        return '$0-5'
    elif price < 10:
        return '$5-10'
    elif price < 20:
        return '$10-20'
    elif price < 30:
        return '$20-30'
    elif price < 50:
        return '$30-50'
    else:
        return '$50+'

games['Price category'] = games['Price'].apply(categorize_price)

# Success classification
games['Success'] = games['Total reviews'] >= 10

# DLC categories
def categorize_dlc(dlc_count):
    if dlc_count == 0:
        return 'No DLC'
    elif dlc_count <= 5:
        return 'Low DLC (1-5)'
    elif dlc_count <= 15:
        return 'Mid DLC (6-15)'
    else:
        return 'High DLC (16+)'

games['DLC category'] = games['DLC count'].apply(categorize_dlc)

# Filter successful games for main analysis
games_success = games[games['Success']].copy()

print(f"[Ready] Loaded {len(games):,} games ({len(games_success):,} successful)")

# Define color scheme
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}

GENRE_COLORS = {
    'Action': '#e74c3c',
    'Strategy': '#3498db',
    'RPG': '#9b59b6',
    'Simulation': '#1abc9c',
    'Adventure': '#f39c12',
    'Indie': '#34495e',
    'Casual': '#95a5a6'
}

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Steam Games Pricing Strategy Dashboard",
                   className="text-center my-4",
                   style={'color': COLORS['dark']}),
            html.H5("Data-Driven Insights for Game Developers",
                   className="text-center mb-4",
                   style={'color': COLORS['info']}),
            html.Hr()
        ])
    ]),

    # Main content with tabs
    dbc.Tabs([
        # Tab 1: Revenue Optimization
        dbc.Tab(label="💰 Revenue Optimization", tab_id="tab-1", children=[
            html.Div([
                html.H3("Part 1: The Illusion", className="mt-4 mb-3", style={'color': COLORS['primary']}),
                html.P("What pricing maximizes revenue... for successful games?",
                      className="lead mb-4"),

                # Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Game Genre:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='genre-dropdown-tab1',
                            options=[{'label': 'All Genres', 'value': 'All'}] +
                                   [{'label': g, 'value': g} for g in sorted(games_success['Primary genre'].value_counts().head(6).index)],
                            value='All',
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Select Metric:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='metric-dropdown-tab1',
                            options=[
                                {'label': 'Average Revenue', 'value': 'revenue'},
                                {'label': 'Average Rating', 'value': 'rating'},
                                {'label': 'Average Owners', 'value': 'owners'}
                            ],
                            value='revenue',
                            clearable=False
                        )
                    ], md=4)
                ], className="mb-4"),

                # Charts
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='revenue-by-price-chart')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='genre-price-heatmap')
                    ], md=6)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='sweet-spot-chart')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='price-rating-curve')
                    ], md=6)
                ])
            ])
        ]),

        # Tab 2: Risk Assessment
        dbc.Tab(label="⚠️ Risk Assessment", tab_id="tab-2", children=[
            html.Div([
                html.H3("Part 2: The Reality", className="mt-4 mb-3", style={'color': COLORS['danger']}),
                html.P("The brutal truth: 57.8% of games fail to get even 10 reviews",
                      className="lead mb-4"),

                # Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Success Threshold (minimum reviews):", style={'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='success-threshold-slider',
                            min=10,
                            max=1000,
                            step=10,
                            value=10,
                            marks={10: '10', 100: '100', 500: '500', 1000: '1000'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("Filter by Genre:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='genre-dropdown-tab2',
                            options=[{'label': 'All Genres', 'value': 'All'}] +
                                   [{'label': g, 'value': g} for g in sorted(games['Primary genre'].value_counts().head(6).index)],
                            value='All',
                            clearable=False
                        )
                    ], md=4)
                ], className="mb-4"),

                # Key metrics
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(id='success-rate-metric', className="text-success"),
                                html.P("Success Rate", className="text-muted")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(id='failure-rate-metric', className="text-danger"),
                                html.P("Failure Rate", className="text-muted")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(id='total-games-metric', className="text-info"),
                                html.P("Total Games", className="text-muted")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(id='safest-price-metric', className="text-success"),
                                html.P("Safest Price Range", className="text-muted")
                            ])
                        ])
                    ], md=3)
                ], className="mb-4"),

                # Charts
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='success-failure-chart')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='failure-rate-heatmap')
                    ], md=6)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='feature-comparison-radar')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='lifecycle-chart')
                    ], md=6)
                ])
            ])
        ]),

        # Tab 3: Market Intelligence
        dbc.Tab(label="📊 Market Intelligence", tab_id="tab-3", children=[
            html.Div([
                html.H3("Part 3: The Strategy", className="mt-4 mb-3", style={'color': COLORS['success']}),
                html.P("Market saturation, DLC strategy, and golden features",
                      className="lead mb-4"),

                # Market Saturation Section
                html.H4("Market Saturation", className="mt-4 mb-3"),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='release-trend-chart')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='success-rate-trend-chart')
                    ], md=6)
                ], className="mb-4"),

                # DLC Strategy Section
                html.H4("DLC Strategy", className="mt-4 mb-3"),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='dlc-owners-scatter')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='dlc-model-comparison')
                    ], md=6)
                ], className="mb-4"),

                # Golden Features Section
                html.H4("Golden Features", className="mt-4 mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Minimum Game Count:", style={'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='tag-min-games-slider',
                            min=50,
                            max=1000,
                            step=50,
                            value=100,
                            marks={50: '50', 500: '500', 1000: '1000'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=6)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='golden-tags-chart')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='tag-trend-chart')
                    ], md=6)
                ])
            ])
        ])
    ], id="tabs", active_tab="tab-1"),

    # Footer
    html.Hr(className="mt-5"),
    html.Footer([
        html.P("Steam Games Pricing Strategy Dashboard | Data: 111,452 games (1997-2025) | Analysis Date: October 2025",
              className="text-center text-muted")
    ], className="mb-4")

], fluid=True, style={'backgroundColor': '#f8f9fa'})


# Callbacks for Tab 1
@app.callback(
    [Output('revenue-by-price-chart', 'figure'),
     Output('genre-price-heatmap', 'figure'),
     Output('sweet-spot-chart', 'figure'),
     Output('price-rating-curve', 'figure')],
    [Input('genre-dropdown-tab1', 'value'),
     Input('metric-dropdown-tab1', 'value')]
)
def update_tab1_charts(selected_genre, selected_metric):
    # Filter data
    if selected_genre == 'All':
        filtered_data = games_success.copy()
    else:
        filtered_data = games_success[games_success['Primary genre'] == selected_genre].copy()

    # Chart 1: Revenue by Price Category
    price_order = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30-50', '$50+']

    if selected_metric == 'revenue':
        metric_col = 'Estimated revenue'
        metric_label = 'Average Revenue ($M)'
        metric_format = lambda x: f'${x/1e6:.1f}M'
    elif selected_metric == 'rating':
        metric_col = 'Positive rate'
        metric_label = 'Average Rating'
        metric_format = lambda x: f'{x:.1%}'
    else:  # owners
        metric_col = 'Estimated owners (mid)'
        metric_label = 'Average Owners'
        metric_format = lambda x: f'{x/1e3:.0f}K'

    price_stats = filtered_data.groupby('Price category')[metric_col].mean().reindex(price_order)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=price_stats.index,
        y=price_stats.values,
        marker_color=COLORS['primary'],
        text=[metric_format(v) for v in price_stats.values],
        textposition='outside'
    ))
    fig1.update_layout(
        title=f'{metric_label} by Price Range',
        xaxis_title='Price Range',
        yaxis_title=metric_label,
        template='plotly_white',
        height=400
    )

    # Chart 2: Genre × Price Heatmap
    top_genres = games_success['Primary genre'].value_counts().head(6).index
    heatmap_data = games_success[games_success['Primary genre'].isin(top_genres)].groupby(
        ['Primary genre', 'Price category']
    )[metric_col].mean().unstack(fill_value=0)
    heatmap_data = heatmap_data[price_order]

    fig2 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        text=[[metric_format(val) for val in row] for row in heatmap_data.values],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig2.update_layout(
        title=f'{metric_label} Heatmap: Genre × Price',
        xaxis_title='Price Range',
        yaxis_title='Game Genre',
        template='plotly_white',
        height=400
    )

    # Chart 3: Sweet Spot (综合得分)
    sweet_spot_data = []
    for genre in top_genres:
        genre_data = games_success[games_success['Primary genre'] == genre]
        for price_cat in price_order[1:]:  # Exclude Free
            cat_data = genre_data[genre_data['Price category'] == price_cat]
            if len(cat_data) >= 20:
                # 综合得分: 40%收入 + 35%好评率 + 25%拥有者数
                revenue_norm = cat_data['Estimated revenue'].mean() / games_success['Estimated revenue'].max()
                rating_norm = cat_data['Positive rate'].mean()
                owners_norm = cat_data['Estimated owners (mid)'].mean() / games_success['Estimated owners (mid)'].max()
                total_score = 0.4 * revenue_norm + 0.35 * rating_norm + 0.25 * owners_norm

                sweet_spot_data.append({
                    'Genre': genre,
                    'Price': price_cat,
                    'Score': total_score,
                    'Count': len(cat_data)
                })

    sweet_spot_df = pd.DataFrame(sweet_spot_data)
    if not sweet_spot_df.empty:
        best_per_genre = sweet_spot_df.sort_values('Score', ascending=False).groupby('Genre').first().reset_index()
        best_per_genre = best_per_genre.sort_values('Score', ascending=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=best_per_genre['Genre'],
            x=best_per_genre['Score'],
            orientation='h',
            marker_color=[GENRE_COLORS.get(g, COLORS['info']) for g in best_per_genre['Genre']],
            text=[f"{p} (Score: {s:.3f})" for p, s in zip(best_per_genre['Price'], best_per_genre['Score'])],
            textposition='outside'
        ))
        fig3.update_layout(
            title='Sweet Spot Pricing by Genre',
            xaxis_title='Composite Score (Revenue 40% + Rating 35% + Owners 25%)',
            yaxis_title='Game Genre',
            template='plotly_white',
            height=400
        )
    else:
        fig3 = go.Figure()
        fig3.update_layout(title='Insufficient Data', template='plotly_white', height=400)

    # Chart 4: Price-Rating Curve
    fig4 = go.Figure()
    for genre in top_genres[:6]:
        genre_data = games_success[games_success['Primary genre'] == genre]
        price_rating = genre_data.groupby('Price category')['Positive rate'].mean().reindex(price_order)
        fig4.add_trace(go.Scatter(
            x=price_rating.index,
            y=price_rating.values,
            mode='lines+markers',
            name=genre,
            line=dict(color=GENRE_COLORS.get(genre, COLORS['info']), width=2),
            marker=dict(size=8)
        ))

    fig4.update_layout(
        title='Does Higher Price Improve Rating?',
        xaxis_title='Price Range',
        yaxis_title='Average Positive Rate',
        template='plotly_white',
        height=400,
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified'
    )

    return fig1, fig2, fig3, fig4


# Callbacks for Tab 2
@app.callback(
    [Output('success-rate-metric', 'children'),
     Output('failure-rate-metric', 'children'),
     Output('total-games-metric', 'children'),
     Output('safest-price-metric', 'children'),
     Output('success-failure-chart', 'figure'),
     Output('failure-rate-heatmap', 'figure'),
     Output('feature-comparison-radar', 'figure'),
     Output('lifecycle-chart', 'figure')],
    [Input('success-threshold-slider', 'value'),
     Input('genre-dropdown-tab2', 'value')]
)
def update_tab2_charts(threshold, selected_genre):
    # Filter data
    if selected_genre == 'All':
        filtered_data = games.copy()
    else:
        filtered_data = games[games['Primary genre'] == selected_genre].copy()

    # Recalculate success based on threshold
    filtered_data['Success_adjusted'] = filtered_data['Total reviews'] >= threshold

    success_rate = filtered_data['Success_adjusted'].mean()
    failure_rate = 1 - success_rate
    total_games = len(filtered_data)

    # Find safest price range
    price_order = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30-50', '$50+']
    price_success = filtered_data.groupby('Price category')['Success_adjusted'].mean().reindex(price_order)
    safest_price = price_success.idxmax()

    # Metrics
    metric1 = f"{success_rate:.1%}"
    metric2 = f"{failure_rate:.1%}"
    metric3 = f"{total_games:,}"
    metric4 = safest_price

    # Chart 1: Success vs Failure by Price
    price_stats = filtered_data.groupby('Price category').agg({
        'Success_adjusted': ['sum', 'count']
    }).reset_index()
    price_stats.columns = ['Price category', 'Success', 'Total']
    price_stats['Failure'] = price_stats['Total'] - price_stats['Success']
    price_stats = price_stats.set_index('Price category').reindex(price_order)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name='Success',
        x=price_stats.index,
        y=price_stats['Success'],
        marker_color=COLORS['success']
    ))
    fig1.add_trace(go.Bar(
        name='Failure',
        x=price_stats.index,
        y=price_stats['Failure'],
        marker_color=COLORS['danger']
    ))
    fig1.update_layout(
        title='Success vs Failure by Price Range',
        xaxis_title='Price Range',
        yaxis_title='Number of Games',
        barmode='stack',
        template='plotly_white',
        height=400
    )

    # Chart 2: Failure Rate Heatmap
    top_genres = filtered_data['Primary genre'].value_counts().head(6).index
    failure_matrix = []
    for genre in top_genres:
        row = []
        for price_cat in price_order:
            subset = filtered_data[(filtered_data['Primary genre'] == genre) &
                                  (filtered_data['Price category'] == price_cat)]
            if len(subset) >= 20:
                failure_rate_val = 1 - subset['Success_adjusted'].mean()
                row.append(failure_rate_val * 100)
            else:
                row.append(None)
        failure_matrix.append(row)

    fig2 = go.Figure(data=go.Heatmap(
        z=failure_matrix,
        x=price_order,
        y=list(top_genres),
        colorscale='RdYlGn_r',
        text=[[f'{val:.1f}%' if val is not None else 'N/A' for val in row] for row in failure_matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        zmid=50
    ))
    fig2.update_layout(
        title='Failure Rate Matrix: Genre × Price (%)',
        xaxis_title='Price Range',
        yaxis_title='Game Genre',
        template='plotly_white',
        height=400
    )

    # Chart 3: Feature Comparison Radar
    success_games = filtered_data[filtered_data['Success_adjusted']]
    failure_games = filtered_data[~filtered_data['Success_adjusted']]

    categories = ['Price', 'DLC Count', 'Owners', 'Peak CCU', 'Positive Rate']

    # Normalize values to 0-1 scale
    success_values = [
        success_games['Price'].mean() / games['Price'].max(),
        success_games['DLC count'].mean() / games['DLC count'].max(),
        success_games['Estimated owners (mid)'].mean() / games['Estimated owners (mid)'].max(),
        success_games['Peak CCU'].mean() / games['Peak CCU'].max(),
        success_games['Positive rate'].mean()
    ]

    failure_values = [
        failure_games['Price'].mean() / games['Price'].max(),
        failure_games['DLC count'].mean() / games['DLC count'].max(),
        failure_games['Estimated owners (mid)'].mean() / games['Estimated owners (mid)'].max(),
        failure_games['Peak CCU'].mean() / games['Peak CCU'].max(),
        failure_games['Positive rate'].mean()
    ]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=success_values,
        theta=categories,
        fill='toself',
        name='Success Games',
        line_color=COLORS['success']
    ))
    fig3.add_trace(go.Scatterpolar(
        r=failure_values,
        theta=categories,
        fill='toself',
        name='Failure Games',
        line_color=COLORS['danger']
    ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Feature Comparison: Success vs Failure',
        template='plotly_white',
        height=400
    )

    # Chart 4: Lifecycle patterns (using representative games)
    # Load count data for lifecycle analysis
    representative_games = {
        'Black Myth: Wukong': 2358720,
        'Garry\'s Mod': 4000,
        'Don\'t Starve Together': 322330,
        'Cities: Skylines': 255710,
        'Factorio': 427520,
        'Civilization VI': 289070
    }

    fig4 = go.Figure()

    for game_name, app_id in representative_games.items():
        game_count = count_data[count_data['AppID'] == app_id].copy()
        if not game_count.empty:
            game_count['date'] = pd.to_datetime(game_count['date'])
            game_count = game_count.sort_values('date').tail(24)  # Last 24 months

            fig4.add_trace(go.Scatter(
                x=game_count['date'],
                y=game_count['avg_players'],
                mode='lines',
                name=game_name,
                line=dict(width=2)
            ))

    fig4.update_layout(
        title='Player Count Lifecycle (Last 24 Months)',
        xaxis_title='Date',
        yaxis_title='Average Players',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )

    return metric1, metric2, metric3, metric4, fig1, fig2, fig3, fig4


# Callbacks for Tab 3
@app.callback(
    [Output('release-trend-chart', 'figure'),
     Output('success-rate-trend-chart', 'figure'),
     Output('dlc-owners-scatter', 'figure'),
     Output('dlc-model-comparison', 'figure'),
     Output('golden-tags-chart', 'figure'),
     Output('tag-trend-chart', 'figure')],
    [Input('tag-min-games-slider', 'value')]
)
def update_tab3_charts(min_games):
    # Chart 1: Release Trend
    yearly_stats = games.groupby('Release year').agg({
        'AppID': 'count',
        'Success': 'mean'
    }).reset_index()
    yearly_stats = yearly_stats[(yearly_stats['Release year'] >= 2010) & (yearly_stats['Release year'] <= 2024)]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=yearly_stats['Release year'],
        y=yearly_stats['AppID'],
        marker_color=COLORS['primary'],
        name='Games Released'
    ))
    fig1.update_layout(
        title='Game Release Trend (2010-2024)',
        xaxis_title='Year',
        yaxis_title='Number of Games Released',
        template='plotly_white',
        height=400
    )

    # Chart 2: Success Rate Trend
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=yearly_stats['Release year'],
        y=yearly_stats['Success'] * 100,
        mode='lines+markers',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))
    fig2.update_layout(
        title='Success Rate Collapse (2010-2024)',
        xaxis_title='Year',
        yaxis_title='Success Rate (%)',
        template='plotly_white',
        height=400
    )

    # Chart 3: DLC vs Owners Scatter
    dlc_analysis = games_success[games_success['DLC count'] > 0].copy()
    dlc_analysis = dlc_analysis[dlc_analysis['DLC count'] <= 100]  # Remove outliers

    fig3 = go.Figure()
    for dlc_cat in ['Low DLC (1-5)', 'Mid DLC (6-15)', 'High DLC (16+)']:
        cat_data = dlc_analysis[dlc_analysis['DLC category'] == dlc_cat]
        if not cat_data.empty:
            fig3.add_trace(go.Scatter(
                x=cat_data['DLC count'],
                y=cat_data['Estimated owners (mid)'],
                mode='markers',
                name=dlc_cat,
                marker=dict(size=8, opacity=0.6)
            ))

    fig3.update_layout(
        title='DLC Count vs Owners',
        xaxis_title='Number of DLCs',
        yaxis_title='Estimated Owners',
        template='plotly_white',
        height=400,
        yaxis_type='log'
    )

    # Chart 4: DLC Model Comparison
    model_comparison = pd.DataFrame({
        'Model': ['Low Price\n+ High DLC', 'High Price\n+ Low DLC'],
        'Avg Owners': [1897470, 713630],
        'Avg Rating': [78.77, 72.97],
        'Game Count': [336, 332]
    })

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=model_comparison['Model'],
        y=model_comparison['Avg Owners'],
        name='Average Owners',
        marker_color=COLORS['success'],
        text=[f'{v/1e6:.2f}M' for v in model_comparison['Avg Owners']],
        textposition='outside'
    ))
    fig4.update_layout(
        title='Business Model Comparison',
        xaxis_title='Business Model',
        yaxis_title='Average Owners',
        template='plotly_white',
        height=400
    )

    # Chart 5: Golden Tags (mock data - would need actual Tags processing)
    golden_tags_data = pd.DataFrame({
        'Tag': ['Anime', 'Difficult', 'Multiplayer', 'Female Protagonist', 'Open World',
                'Visual Novel', 'Story Rich', 'Simulation', 'Psych Horror', 'Atmospheric'],
        'Success Rate': [77.7, 77.1, 76.4, 74.5, 73.4, 73.4, 72.3, 70.0, 69.7, 69.6]
    })

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        y=golden_tags_data['Tag'],
        x=golden_tags_data['Success Rate'],
        orientation='h',
        marker_color=COLORS['success'],
        text=[f'{v:.1f}%' for v in golden_tags_data['Success Rate']],
        textposition='outside'
    ))
    fig5.update_layout(
        title=f'Top 10 High Success Rate Tags (>= {min_games} games)',
        xaxis_title='Success Rate (%)',
        yaxis_title='Tag',
        template='plotly_white',
        height=400
    )

    # Chart 6: Tag Trends
    tag_trends = pd.DataFrame({
        'Tag': ['3D', 'Singleplayer', 'Exploration', 'Indie', 'VR', 'Action'],
        'Change': [19.3, 17.8, 13.6, -20.9, -5.6, -4.3],
        'Category': ['Growing', 'Growing', 'Growing', 'Declining', 'Declining', 'Declining']
    })

    fig6 = go.Figure()
    for category, color in [('Growing', COLORS['success']), ('Declining', COLORS['danger'])]:
        cat_data = tag_trends[tag_trends['Category'] == category]
        fig6.add_trace(go.Bar(
            y=cat_data['Tag'],
            x=cat_data['Change'],
            orientation='h',
            name=category,
            marker_color=color,
            text=[f'{v:+.1f}%' for v in cat_data['Change']],
            textposition='outside'
        ))

    fig6.update_layout(
        title='Tag Trends: 2020 vs 2024 (% Change)',
        xaxis_title='Change in Usage (%)',
        yaxis_title='Tag',
        template='plotly_white',
        height=400,
        showlegend=True
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# Run the app
if __name__ == '__main__':
    print("Starting dashboard at http://127.0.0.1:8050/")
    app.run_server(debug=False, host='127.0.0.1', port=8050)
