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

# Helper function to apply Steam dark theme to figures
def apply_steam_theme(fig):
    """Apply Steam-style dark theme to plotly figures"""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1b2838',
        plot_bgcolor='#171a21',
        font={'color': "#fdfcf9", 'family': 'Arial, sans-serif'},
        title_font={'color': "#ffffff"},
        legend=dict(
            bgcolor='rgba(27, 40, 56, 0.8)',
            bordercolor='#66c0f4',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='#1b2838',
            font_color="#ffffff",
            bordercolor='#66c0f4'
        )
    )
    return fig

# Define color scheme - Steam style
COLORS = {
    'primary': '#66c0f4',  # Steam blue (neutral/info)
    'success': "#12f33b",  # Green (success/low risk/positive)
    'warning': "#FFFF00",  # Yellow (warning/medium risk)
    'danger': "#d92b18",   # Red (failure/high risk/negative)
    'info': "#8266f4",     # Purple (information)
    'dark': '#171a21',     # Steam dark background
    'light': "#ffffff",    # Steam light text
    'bg_dark': '#1b2838',  # Steam secondary dark
    'bg_gradient': '#2a475e'  # Steam gradient blue
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
    # Header with Steam style
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src='./assets/logo_steam.svg', 
                        style={'height': '80px', 'marginBottom': '20px'}),
                html.H1("Steam Games Pricing Strategy Dashboard",
                       className="text-center my-4",
                       style={'color': COLORS['light'], 'fontWeight': 'bold'}),
                html.H5("Data-Driven Insights for Game Developers",
                       className="text-center mb-4",
                       style={'color': COLORS['primary']}),
            ], style={
                'backgroundColor': COLORS['dark'],
                'padding': '30px',
                'borderRadius': '10px',
                'marginBottom': '20px',
                'backgroundImage': f'linear-gradient(to right, {COLORS["dark"]}, {COLORS["bg_gradient"]})'
            }),
            html.Hr(style={'borderColor': COLORS['bg_gradient'], 'borderWidth': '2px'})
        ])
    ]),

    # Main content with tabs
    dbc.Tabs([

        # Tab 0: Overview
        dbc.Tab(label="🎮 Steam Game Overview", tab_id="tab-0", 
                label_style={'color': COLORS['light']}, 
                active_label_style={'backgroundColor': COLORS['primary'], 'color': COLORS['dark']},
                children=[
            html.Div([
                html.H3("Overview of Steam market", className="mt-4 mb-3", style={'color': COLORS['primary'], 'textAlign': 'center'}),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='release-density-line'), 
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='price-owners-scatter'),
                    ], md=6),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='platform-pie'),
                    ], md=4),
                    dbc.Col([
                        dcc.Graph(id='top-publishers-bar'),
                    ], md=4),
                    dbc.Col([
                        dcc.Graph(id='genre-bubble'),
                    ], md=4)

                ], className="mb-4"),
                
            ], style={'backgroundColor': COLORS['bg_dark'], 'padding': '20px', 'borderRadius': '10px'})
        ]),
        # Tab 1: Revenue Optimization
        dbc.Tab(label="💰 Revenue Optimization", tab_id="tab-1",
                label_style={'color': COLORS['light']},
                active_label_style={'backgroundColor': COLORS['primary'], 'color': COLORS['dark']},
                children=[
            html.Div([
                html.H3("Part 1: The Illusion", className="mt-4 mb-3", style={'color': COLORS['primary'], 'textAlign': 'center'}),
                html.P("What pricing maximizes revenue... for successful games?",
                      className="lead mb-4", style={'color': COLORS['light'], 'textAlign': 'center'}),

                # Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Game Genre:", style={'fontWeight': 'bold', 'color': COLORS['light']}),
                        dcc.Dropdown(
                            id='genre-dropdown-tab1',
                            options=[{'label': 'All Genres', 'value': 'All'}] +
                                   [{'label': g, 'value': g} for g in sorted(games_success['Primary genre'].value_counts().head(6).index)],
                            value='All',
                            clearable=False,
                            style={'color': '#000000'}  # Black text for readability
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Select Metric:", style={'fontWeight': 'bold', 'color': COLORS['light']}),
                        dcc.Dropdown(
                            id='metric-dropdown-tab1',
                            options=[
                                {'label': 'Average Revenue', 'value': 'revenue'},
                                {'label': 'Average Rating', 'value': 'rating'},
                                {'label': 'Average Owners', 'value': 'owners'}
                            ],
                            value='revenue',
                            clearable=False,
                            style={'color': '#000000'}  # Black text for readability
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
            ], style={'backgroundColor': COLORS['bg_dark'], 'padding': '20px', 'borderRadius': '10px'})
        ]),

        # Tab 2: Risk Assessment
        dbc.Tab(label="⚠️ Risk Assessment", tab_id="tab-2",
                label_style={'color': COLORS['light']},
                active_label_style={'backgroundColor': COLORS['primary'], 'color': COLORS['dark']},
                children=[
            html.Div([
                html.H3("Part 2: The Reality", className="mt-4 mb-3", style={'color': COLORS['danger'], 'textAlign': 'center'}),
                html.P("The brutal truth: 57.8% of games fail to get even 10 reviews",
                      className="lead mb-4", style={'color': COLORS['light'], 'textAlign': 'center'}),

                # Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Success Threshold (minimum reviews):", style={'fontWeight': 'bold', 'color': COLORS['light']}),
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
                        html.Label("Filter by Genre:", style={'fontWeight': 'bold', 'color': COLORS['light']}),
                        dcc.Dropdown(
                            id='genre-dropdown-tab2',
                            options=[{'label': 'All Genres', 'value': 'All'}] +
                                   [{'label': g, 'value': g} for g in sorted(games['Primary genre'].value_counts().head(6).index)],
                            value='All',
                            clearable=False,
                            style={'color': '#000000'}  # Black text for readability
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
            ], style={'backgroundColor': COLORS['bg_dark'], 'padding': '20px', 'borderRadius': '10px'})
        ]),

        # Tab 3: Market Intelligence
        dbc.Tab(label="📊 Market Intelligence", tab_id="tab-3",
                label_style={'color': COLORS['light']},
                active_label_style={'backgroundColor': COLORS['primary'], 'color': COLORS['dark']},
                children=[
            html.Div([
                html.H3("Part 3: The Strategy", className="mt-4 mb-3", style={'color': COLORS['success'], 'textAlign': 'center'}),
                html.P("Market saturation, DLC strategy, and golden features",
                      className="lead mb-4", style={'color': COLORS['light'], 'textAlign': 'center'}),

                # Market Saturation Section
                html.H4("Market Saturation", className="mt-4 mb-3", style={'color': COLORS['primary']}),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='release-trend-chart')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='success-rate-trend-chart')
                    ], md=6)
                ], className="mb-4"),

                # DLC Strategy Section
                html.H4("DLC Strategy", className="mt-4 mb-3", style={'color': COLORS['primary']}),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='dlc-owners-scatter')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='dlc-model-comparison')
                    ], md=6)
                ], className="mb-4"),

                # Golden Features Section
                html.H4("Golden Features", className="mt-4 mb-3", style={'color': COLORS['primary']}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Minimum Game Count:", style={'fontWeight': 'bold', 'color': COLORS['light']}),
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
            ], style={'backgroundColor': COLORS['bg_dark'], 'padding': '20px', 'borderRadius': '10px'})
        ]),

        # Tab 4: Case Study
        dbc.Tab(label="🎯 Case Study", tab_id="tab-4",
                label_style={'color': COLORS['light']},
                active_label_style={'backgroundColor': COLORS['primary'], 'color': COLORS['dark']},
                children=[
            html.Div([
                html.H3("Case Study: Monster Hunter: World", className="mt-4 mb-3",
                       style={'color': COLORS['primary'], 'textAlign': 'center'}),
                html.P("Real-world evidence: Mid-price + High DLC strategy success",
                      className="lead mb-4", style={'color': COLORS['light'], 'textAlign': 'center'}),

                # Key Metrics Cards
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Base Price", style={'color': '#c0c0c0', 'fontSize': '14px'}),
                                html.H3("$29.99", style={'color': COLORS['primary']}),
                                html.P("Mid-price sweet spot", className="mb-0", style={'color': COLORS['light'], 'fontSize': '13px'})
                            ])
                        ], style={'backgroundColor': COLORS['dark'], 'borderColor': COLORS['primary']})
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("DLC Count", style={'color': '#c0c0c0', 'fontSize': '14px'}),
                                html.H3("200", style={'color': COLORS['success']}),
                                html.P("Extensive content library", className="mb-0", style={'color': COLORS['light'], 'fontSize': '13px'})
                            ])
                        ], style={'backgroundColor': COLORS['dark'], 'borderColor': COLORS['success']})
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Estimated Owners", style={'color': '#c0c0c0', 'fontSize': '14px'}),
                                html.H3("15.0M", style={'color': COLORS['warning']}),
                                html.P("Top 0.3% in price range", className="mb-0", style={'color': COLORS['light'], 'fontSize': '13px'})
                            ])
                        ], style={'backgroundColor': COLORS['dark'], 'borderColor': COLORS['warning']})
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("User Rating", style={'color': '#c0c0c0', 'fontSize': '14px'}),
                                html.H3("85.5%", style={'color': COLORS['info']}),
                                html.P("Highly positive reviews", className="mb-0", style={'color': COLORS['light'], 'fontSize': '13px'})
                            ])
                        ], style={'backgroundColor': COLORS['dark'], 'borderColor': COLORS['info']})
                    ], md=3)
                ], className="mb-4"),

                # Charts Row 1
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='case-strategy-comparison')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='case-game-comparison')
                    ], md=6)
                ], className="mb-4"),

                # Charts Row 2
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='case-price-history')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='case-player-trend')
                    ], md=6)
                ], className="mb-4"),

                # Charts Row 3
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='case-radar-chart')
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id='case-dlc-scatter')
                    ], md=6)
                ])
            ], style={'backgroundColor': COLORS['bg_dark'], 'padding': '20px', 'borderRadius': '10px'})
        ])
    ], id="tabs", active_tab="tab-1", style={'backgroundColor': COLORS['dark']}),

    # Footer
    html.Hr(className="mt-5", style={'borderColor': COLORS['bg_gradient'], 'borderWidth': '2px'}),
    html.Footer([
        html.P("Steam Games Pricing Strategy Dashboard | Data: 111,452 games (1997-2025) | Analysis Date: October 2025",
              className="text-center", style={'color': COLORS['light']})
    ], className="mb-4")

], fluid=True, style={
    'backgroundColor': COLORS['bg_dark'],
    'backgroundImage': f'linear-gradient(to bottom, {COLORS["dark"]}, {COLORS["bg_dark"]})',
    'minHeight': '100vh',
    'color': COLORS['light']
})

# Callbacks for Tab 0

@app.callback(
    [Output('platform-pie', 'figure'),
     Output('price-owners-scatter', 'figure'),
     Output('top-publishers-bar', 'figure'),
     Output('genre-bubble', 'figure'),
     Output('release-density-line', 'figure')],
    Input('tabs', 'active_tab')
)
def update_tab1_charts(active_tab):


    # =================== 图1：平台支持分布 ===================
    try:
    # 统计每个平台支持的游戏数量（去重：一个游戏支持多个平台）
        platform_counts = games_success[['Windows', 'Mac', 'Linux']].sum()
        total_games = len(games_success)

        # 构建饼图数据
        fig1 = go.Figure(data=[go.Pie(
            labels=platform_counts.index,
            values=platform_counts.values,
            hole=0.4,  # 环形饼图，更美观
            marker_colors=[COLORS['primary'], COLORS['warning'], COLORS['info']],
            textinfo='label+percent',
            textposition='inside',
            hovertemplate=
                '<b>%{label}</b><br>'
                'Games: %{value:,}<br>'
                'Share: %{percent}<extra></extra>'
        )])

        fig1.update_layout(
            title={
                'text': 'Platform Support Distribution',
                'font': {'size': 14, 'color': COLORS['light']},
                'x': 0.5,
                'xanchor': 'center'
            },
            template='plotly_dark',
            paper_bgcolor=COLORS['dark'],
            plot_bgcolor=COLORS['bg_dark'],
            font={'color': COLORS['light']},
            height=420,
            autosize=False,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )
    except Exception as e:
        fig1 = go.Figure().add_annotation(
            text="Platform Data Unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # =================== 图2：价格 vs 拥有者散点图 ===================
    try:
        df2 = games_success[['Name', 'Price', 'Estimated owners (mid)']].copy()
        df2 = df2[df2['Estimated owners (mid)'] > 0]

        fig2 = px.scatter(
            df2, 
            x='Price', 
            y='Estimated owners (mid)',
            hover_name='Name',
            log_y=True,
            color_discrete_sequence=[COLORS['primary']]
        )
        fig2.update_traces(marker=dict(size=5, opacity=0.7))
        
        apply_steam_theme(fig2)
        fig2.update_layout(
            title={
                'text': 'Price vs Estimated Owners',
                'font': {'size': 14},
                'x': 0.5
            },
            xaxis_title='Price (USD)',
            yaxis_title='Estimated Owners (Log Scale)',
            height=420,
            autosize=False,
            margin=dict(l=40, r=20, t=60, b=40),
            showlegend=False
        )
    except:
        fig2 = go.Figure().add_annotation(text="Scatter Data Unavailable", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # =================== 图3：Top 10 发行商（新增） ===================
    try:
        # 按总拥有者数量统计发行商（使用成功游戏）
        publisher_stats = games_success.groupby('Publishers')['Estimated owners (mid)'].sum().sort_values(ascending=False).head(10)
        publisher_stats = publisher_stats.reset_index()
        
        # 格式化数字（百万单位）
        publisher_stats['Formatted Owners'] = publisher_stats['Estimated owners (mid)'].apply(
            lambda x: f'{x/1e6:.1f}M'
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=publisher_stats['Publishers'],
            x=publisher_stats['Estimated owners (mid)'],
            orientation='h',
            marker_color=COLORS['success'],
            text=publisher_stats['Formatted Owners'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Total Owners: %{x:,.0f}<br>' +
                          '%{text}<extra></extra>'
        ))

        apply_steam_theme(fig3)
        fig3.update_layout(
            title={
                'text': 'Top 10 Publishers by Total Estimated Owners',
                'font': {'size': 14},
                'x': 0.5
            },
            xaxis_title='Total Estimated Owners',
            yaxis_title='Publisher',
            height=420,
            autosize=False,
            margin=dict(l=200, r=20, t=60, b=40),  # l=200 给长出版社名留空间
            showlegend=False
        )
    except:
        fig3 = go.Figure().add_annotation(text="Publisher Data Unavailable", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)


    # =================== 图4：Genre × Owners 热力图 ===================
    try:
        # --- 步骤1：自动识别列名 ---
        print("正在加载气泡图数据...")  # 调试信息

        # 自动找列
        genre_col = None
        owner_col = None
        appid_col = 'AppID'

        for col in games_success.columns:
            if 'genre' in col.lower():
                genre_col = col
            if 'owner' in col.lower():
                owner_col = col

        if not genre_col or not owner_col:
            raise ValueError("未找到 genre 或 owner 列")

        print(f"使用列: Genre='{genre_col}', Owners='{owner_col}'")

        # --- 步骤2：数据清洗 ---
        df = games[[genre_col, owner_col, appid_col]].copy()
        df = df.dropna(subset=[genre_col, owner_col])
        df[owner_col] = pd.to_numeric(df[owner_col], errors='coerce')
        df = df.dropna(subset=[owner_col])
        df = df[df[owner_col] > 0]

        if df.empty:
            raise ValueError("数据清洗后为空")

        # --- 步骤3：聚合 ---
        genre_stats = df.groupby(genre_col).agg(
            total_owners=(owner_col, 'sum'),
            game_count=(appid_col, 'count'),
            avg_owners=(owner_col, 'mean')
        ).reset_index()

        # 至少 20 款游戏
        genre_stats = genre_stats[genre_stats['game_count'] >= 20]

        # 至少有 3 个类型
        if len(genre_stats) < 3:
            raise ValueError("有效类型少于3个")

        # 取前 12 个（避免拥挤）
        genre_stats = genre_stats.sort_values('total_owners', ascending=False).head(12)

        # --- 步骤4：气泡大小 ---
        min_total = genre_stats['total_owners'].min()
        max_total = genre_stats['total_owners'].max()
        if max_total == min_total:
            genre_stats['bubble_size'] = 50
        else:
            genre_stats['bubble_size'] = 20 + 80 * (genre_stats['total_owners'] - min_total) / (max_total - min_total)

        # --- 步骤5：绘图 ---
        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=genre_stats['game_count'],
            y=genre_stats['avg_owners'],
            mode='markers+text',
            text=genre_stats[genre_col],
            textposition='middle center',
            textfont=dict(color='white', size=9, family='Arial'),
            marker=dict(
                colorscale='Viridis',
                size=genre_stats['bubble_size'],
                color=genre_stats['total_owners'],
                showscale=True,
                colorbar=dict(title="Total Owners", x=1.02),
                line=dict(width=1, color='black')
            ),
            hovertemplate=
                '<b>%{text}</b><br>'
                'Games: %{x:,}<br>'
                'Avg Owners: %{y:,.0f}<br>'
                'Total: %{marker.color:,.0f}<extra></extra>'
        ))

        apply_steam_theme(fig4)
        fig4.update_layout(
            title={
                'text': 'Genre Popularity Bubble Map',
                'font': {'size': 14},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Number of Games',
            yaxis_title='Avg Owners per Game',
            xaxis=dict(type='log', tickformat=',.0f'),
            yaxis=dict(type='log', tickformat=',.0f'),
            height=420,
            autosize=False,
            margin=dict(l=50, r=120, t=60, b=50),
            showlegend=False
        )

        print(f"气泡图成功生成！共 {len(genre_stats)} 个类型")

    except Exception as e:
        print(f"气泡图失败: {str(e)}")
        fig4 = go.Figure()
        fig4.add_annotation(
            text=f"Bubble Map Error<br>{str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,0,0,0.1)"
        )
        # =================== 图5：游戏发行密度折线图 ===================
    try:
        # 按年统计游戏数量
        yearly_releases = games.groupby('Release year').size().reset_index(name='Game Count')
        yearly_releases = yearly_releases[
            (yearly_releases['Release year'] >= 2010) & 
            (yearly_releases['Release year'] <= 2024)
        ].copy()
        yearly_releases['Release year'] = yearly_releases['Release year'].astype(int)

        fig5 = go.Figure()

        # 主折线
        fig5.add_trace(go.Scatter(
            x=yearly_releases['Release year'],
            y=yearly_releases['Game Count'],
            mode='lines+markers',
            name='Releases',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate=
                '<b>Year:</b> %{x}<br>'
                '<b>Games Released:</b> %{y:,}<extra></extra>'
        ))

        # 可选：添加峰值标注
        peak_year = yearly_releases.loc[yearly_releases['Game Count'].idxmax()]
        fig5.add_annotation(
            x=peak_year['Release year'],
            y=peak_year['Game Count'],
            text=f"Peak: {peak_year['Game Count']:,}",
            showarrow=True,
            arrowhead=2,
            ax=20, ay=-30,
            bgcolor="white",
            bordercolor=COLORS['primary'],
            borderwidth=1
        )

        apply_steam_theme(fig5)
        fig5.update_layout(
            title={
                'text': 'Game Release Trend (2010-2024)',
                'font': {'size': 14},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Year',
            yaxis_title='Number of Games Released',
            height=420,
            autosize=False,
            margin=dict(l=40, r=20, t=60, b=40),
            hovermode='x unified'
        )
    except Exception as e:
        fig5 = go.Figure().add_annotation(
            text="Release Data Unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    return fig1, fig2, fig3, fig4, fig5

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
    apply_steam_theme(fig1)
    fig1.update_layout(
        title=f'{metric_label} by Price Range',
        xaxis_title='Price Range',
        yaxis_title=metric_label,
        height=400
    )

    # Chart 2: Genre × Price Heatmap
    # Filter out Unknown genres
    games_success_no_unknown = games_success[games_success['Primary genre'] != 'Unknown'].copy()
    top_genres = games_success_no_unknown['Primary genre'].value_counts().head(6).index
    heatmap_data = games_success_no_unknown[games_success_no_unknown['Primary genre'].isin(top_genres)].groupby(
        ['Primary genre', 'Price category']
    )[metric_col].mean().unstack(fill_value=0)
    heatmap_data = heatmap_data[price_order]

    fig2 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        # colorscale='RdYlGn',
        text=[[metric_format(val) for val in row] for row in heatmap_data.values],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    apply_steam_theme(fig2)
    fig2.update_layout(
        title=f'{metric_label} Heatmap: Genre × Price',
        xaxis_title='Price Range',
        yaxis_title='Game Genre',
        height=400
    )

    # Chart 3: Sweet Spot (综合得分)
    sweet_spot_data = []
    for genre in top_genres:
        genre_data = games_success_no_unknown[games_success_no_unknown['Primary genre'] == genre]
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
        apply_steam_theme(fig3)
        fig3.update_layout(
            title='Sweet Spot Pricing by Genre',
            xaxis_title='Composite Score (Revenue 40% + Rating 35% + Owners 25%)',
            yaxis_title='Game Genre',
            height=400
        )
    else:
        fig3 = go.Figure()
        apply_steam_theme(fig3)
        fig3.update_layout(title='Insufficient Data', height=400)

    # Chart 4: Price-Rating Curve
    fig4 = go.Figure()
    for genre in top_genres[:6]:
        genre_data = games_success_no_unknown[games_success_no_unknown['Primary genre'] == genre]
        price_rating = genre_data.groupby('Price category')['Positive rate'].mean().reindex(price_order)
        fig4.add_trace(go.Scatter(
            x=price_rating.index,
            y=price_rating.values,
            mode='lines+markers',
            name=genre,
            line=dict(color=GENRE_COLORS.get(genre, COLORS['info']), width=2),
            marker=dict(size=8)
        ))

    apply_steam_theme(fig4)
    fig4.update_layout(
        title='Does Higher Price Improve Rating?',
        xaxis_title='Price Range',
        yaxis_title='Average Positive Rate',
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
    apply_steam_theme(fig1)
    fig1.update_layout(
        title='Success vs Failure by Price Range',
        xaxis_title='Price Range',
        yaxis_title='Number of Games',
        barmode='stack',
        height=400
    )

    # Chart 2: Failure Rate Heatmap
    # Filter out Unknown genres
    filtered_data_no_unknown = filtered_data[filtered_data['Primary genre'] != 'Unknown'].copy()
    top_genres = filtered_data_no_unknown['Primary genre'].value_counts().head(6).index
    failure_matrix = []
    for genre in top_genres:
        row = []
        for price_cat in price_order:
            subset = filtered_data_no_unknown[(filtered_data_no_unknown['Primary genre'] == genre) &
                                  (filtered_data_no_unknown['Price category'] == price_cat)]
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
        colorscale='Viridis',
        # colorscale='RdYlGn_r',
        text=[[f'{val:.1f}%' if val is not None else '' for val in row] for row in failure_matrix],
        texttemplate='%{text}',
        textfont={"size": 10, "color": "white"},
        zmid=50
    ))
    apply_steam_theme(fig2)
    fig2.update_layout(
        title='Failure Rate Matrix: Genre × Price (%)',
        xaxis_title='Price Range',
        yaxis_title='Game Genre',
        height=400
    )

    # Chart 3: Price × DLC Count Risk Heatmap
    # Create DLC categories
    dlc_categories = ['No DLC', '1-5 DLC', '6-15 DLC', '16+ DLC']

    def categorize_dlc_simple(count):
        if count == 0:
            return 'No DLC'
        elif count <= 5:
            return '1-5 DLC'
        elif count <= 15:
            return '6-15 DLC'
        else:
            return '16+ DLC'

    filtered_data['DLC_category'] = filtered_data['DLC count'].apply(categorize_dlc_simple)

    # Build risk matrix
    risk_matrix = []
    for dlc_cat in dlc_categories:
        row = []
        for price_cat in price_order:
            subset = filtered_data[
                (filtered_data['DLC_category'] == dlc_cat) &
                (filtered_data['Price category'] == price_cat)
            ]
            if len(subset) >= 10:  # At least 10 games
                failure_rate = (1 - subset['Success_adjusted'].mean()) * 100
                row.append(failure_rate)
            else:
                row.append(None)
        risk_matrix.append(row)

    # Create heatmap
    fig3 = go.Figure(data=go.Heatmap(
        z=risk_matrix,
        x=price_order,
        y=dlc_categories,
        colorscale='RdYlGn_r',  # Red (high risk) to Green (low risk), reversed
        text=[[f'{val:.1f}%' if val is not None else '' for val in row] for row in risk_matrix],
        texttemplate='%{text}',
        textfont={"size": 10, "color": "white"},
        hovertemplate='DLC: %{y}<br>Price: %{x}<br>Failure Rate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Failure<br>Rate (%)")
    ))

    apply_steam_theme(fig3)
    fig3.update_layout(
        title='Risk Matrix: Failure Rate by Price × DLC Strategy',
        xaxis_title='Price Range',
        yaxis_title='DLC Strategy',
        height=400
    )

    # Chart 4: Market Risk Trend Over Years
    # Calculate failure rate by release year
    year_risk_data = []
    for year in range(2010, 2025):
        year_data = filtered_data[filtered_data['Release year'] == year]
        if len(year_data) >= 50:  # At least 50 games per year
            failure_rate = (1 - year_data['Success_adjusted'].mean()) * 100
            game_count = len(year_data)
            year_risk_data.append({
                'Year': year,
                'Failure Rate': failure_rate,
                'Game Count': game_count
            })

    year_risk_df = pd.DataFrame(year_risk_data)

    fig4 = go.Figure()

    # Failure rate line
    fig4.add_trace(go.Scatter(
        x=year_risk_df['Year'],
        y=year_risk_df['Failure Rate'],
        mode='lines+markers',
        name='Failure Rate',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(217, 43, 24, 0.2)',
        hovertemplate='Year: %{x}<br>Failure Rate: %{y:.1f}%<extra></extra>',
        yaxis='y'
    ))

    # Game count bars (secondary axis)
    fig4.add_trace(go.Bar(
        x=year_risk_df['Year'],
        y=year_risk_df['Game Count'],
        name='Games Released',
        marker_color=COLORS['primary'],
        opacity=0.3,
        hovertemplate='Year: %{x}<br>Games: %{y:,}<extra></extra>',
        yaxis='y2'
    ))

    apply_steam_theme(fig4)
    fig4.update_layout(
        title='Market Risk Trend: Failure Rate Over Years (2010-2024)',
        xaxis_title='Release Year',
        yaxis=dict(
            title='Failure Rate (%)',
            side='left',
            range=[0, 100],
            ticksuffix='%'
        ),
        yaxis2=dict(
            title='Games Released',
            overlaying='y',
            side='right'
        ),
        height=400,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(27, 40, 56, 0.8)')
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
    apply_steam_theme(fig1)
    fig1.update_layout(
        title='Game Release Trend (2010-2024)',
        xaxis_title='Year',
        yaxis_title='Number of Games Released',
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
    apply_steam_theme(fig2)
    fig2.update_layout(
        title='Success Rate Collapse (2010-2024)',
        xaxis_title='Year',
        yaxis_title='Success Rate (%)',
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

    apply_steam_theme(fig3)
    fig3.update_layout(
        title='DLC Count vs Owners',
        xaxis_title='Number of DLCs',
        yaxis_title='Estimated Owners',
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
    apply_steam_theme(fig4)
    fig4.update_layout(
        title='Business Model Comparison',
        xaxis_title='Business Model',
        yaxis_title='Average Owners',
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
    apply_steam_theme(fig5)
    fig5.update_layout(
        title=f'Top 10 High Success Rate Tags (>= {min_games} games)',
        xaxis_title='Success Rate (%)',
        yaxis_title='Tag',
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

    apply_steam_theme(fig6)
    fig6.update_layout(
        title='Tag Trends: 2020 vs 2024 (% Change)',
        xaxis_title='Change in Usage (%)',
        yaxis_title='Tag',
        height=400,
        showlegend=True
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# Callbacks for Tab 4 (Case Study)
@app.callback(
    [Output('case-strategy-comparison', 'figure'),
     Output('case-game-comparison', 'figure'),
     Output('case-price-history', 'figure'),
     Output('case-player-trend', 'figure'),
     Output('case-radar-chart', 'figure'),
     Output('case-dlc-scatter', 'figure')],
    [Input('tabs', 'active_tab')]
)
def update_tab4_charts(active_tab):
    # MHW AppID
    MHW_APPID = 582010

    # Extract MHW data
    mhw = games[games['AppID'] == MHW_APPID].iloc[0]
    owners_mid = extract_owners_mid(mhw['Estimated owners'])
    positive_rate = mhw['Positive'] / (mhw['Positive'] + mhw['Negative'])

    # =================== Chart 1: DLC Strategy Comparison ===================
    same_price_range = games[
        (games['Price'] >= 20) & (games['Price'] <= 40) &
        (games['Total reviews'] >= 100)
    ].copy()

    strategies = {
        'Low Price + High DLC\n(MHW Model)': same_price_range[
            (same_price_range['Price'] < 35) & (same_price_range['DLC count'] >= 15)
        ],
        'Mid Price + Mid DLC': same_price_range[
            (same_price_range['Price'] >= 25) & (same_price_range['Price'] < 35) &
            (same_price_range['DLC count'] >= 5) & (same_price_range['DLC count'] < 15)
        ],
        'High Price + Low DLC': same_price_range[
            (same_price_range['Price'] >= 35) & (same_price_range['DLC count'] < 10)
        ]
    }

    strategy_data = []
    for strategy_name, strategy_games in strategies.items():
        if len(strategy_games) > 0:
            strategy_data.append({
                'Strategy': strategy_name,
                'Avg Owners': strategy_games['Estimated owners (mid)'].mean(),
                'Avg Rating': strategy_games['Positive rate'].mean() * 100,
                'Game Count': len(strategy_games)
            })

    strategy_df = pd.DataFrame(strategy_data)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=strategy_df['Strategy'],
        y=strategy_df['Avg Owners'],
        marker_color=[COLORS['success'], COLORS['warning'], COLORS['danger']],
        text=[f"{v/1e6:.2f}M" for v in strategy_df['Avg Owners']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Owners: %{y:,.0f}<br>%{text}<extra></extra>'
    ))

    apply_steam_theme(fig1)
    fig1.update_layout(
        title='DLC Strategy Comparison: Avg Owners by Business Model',
        xaxis_title='Business Model',
        yaxis_title='Average Owners',
        height=400
    )

    # =================== Chart 2: MHW vs Competitors ===================
    comparison_games_list = [
        ('Monster Hunter: World', 582010),
        ('ARK: Survival Evolved', 346110),
        ('Cities: Skylines', 255710),
        ('Beat Saber', 620980),
        ('Dying Light', 239140)
    ]

    comparison_data = []
    for game_name, app_id in comparison_games_list:
        game_data = games[games['AppID'] == app_id]
        if len(game_data) > 0:
            game = game_data.iloc[0]
            comparison_data.append({
                'Game': game_name,
                'Price': game['Price'],
                'DLC Count': game['DLC count'],
                'Owners': extract_owners_mid(game['Estimated owners']),
                'Rating': game['Positive'] / (game['Positive'] + game['Negative']) * 100
            })

    comparison_df = pd.DataFrame(comparison_data)

    fig2 = go.Figure()

    # Owners bar chart
    fig2.add_trace(go.Bar(
        name='Owners (Millions)',
        x=comparison_df['Game'],
        y=comparison_df['Owners'] / 1e6,
        marker_color=COLORS['primary'],
        yaxis='y',
        text=[f"{v/1e6:.1f}M" for v in comparison_df['Owners']],
        textposition='outside'
    ))

    # Rating line chart
    fig2.add_trace(go.Scatter(
        name='Rating (%)',
        x=comparison_df['Game'],
        y=comparison_df['Rating'],
        marker_color=COLORS['success'],
        yaxis='y2',
        mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=10)
    ))

    apply_steam_theme(fig2)
    fig2.update_layout(
        title='MHW vs Similar Games: Owners & Rating Comparison',
        xaxis_title='Game',
        yaxis=dict(title='Owners (Millions)', side='left'),
        yaxis2=dict(title='Rating (%)', overlaying='y', side='right', range=[70, 100]),
        height=400,
        legend=dict(x=0.7, y=1.1, orientation='h')
    )

    # =================== Chart 3: Price History ===================
    mhw_price_history = price_data[price_data['AppID'] == MHW_APPID].copy()
    mhw_price_history['date'] = pd.to_datetime(mhw_price_history['date'])
    mhw_price_history = mhw_price_history.sort_values('date')

    if len(mhw_price_history) > 0:
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=mhw_price_history['date'],
            y=mhw_price_history['price'],
            mode='lines',
            name='Price',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 192, 244, 0.2)',
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        fig3.add_hline(
            y=mhw['Price'],
            line_dash="dash",
            line_color=COLORS['danger'],
            annotation_text=f"Original: ${mhw['Price']:.2f}",
            annotation_position="right"
        )

        apply_steam_theme(fig3)
        fig3.update_layout(
            title='MHW Price History: Discount Strategy Analysis',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400,
            hovermode='x unified'
        )
    else:
        fig3 = go.Figure()
        apply_steam_theme(fig3)
        fig3.update_layout(title='Price History Not Available', height=400)

    # =================== Chart 4: Player Count Trend ===================
    mhw_count_history = count_data[count_data['AppID'] == MHW_APPID].copy()
    mhw_count_history['date'] = pd.to_datetime(mhw_count_history['date'])
    mhw_count_history = mhw_count_history.sort_values('date')

    if len(mhw_count_history) > 0:
        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=mhw_count_history['date'],
            y=mhw_count_history['avg_players'],
            mode='lines',
            name='Avg Players',
            line=dict(color=COLORS['success'], width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 255, 0, 0.2)',
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Avg Players: %{y:,.0f}<extra></extra>'
        ))

        # Mark peak
        peak_idx = mhw_count_history['avg_players'].idxmax()
        peak_data = mhw_count_history.loc[peak_idx]

        fig4.add_annotation(
            x=peak_data['date'],
            y=peak_data['avg_players'],
            text=f"Peak: {peak_data['avg_players']:,.0f}",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-30,
            bgcolor=COLORS['bg_dark'],
            bordercolor=COLORS['success'],
            borderwidth=2
        )

        apply_steam_theme(fig4)
        fig4.update_layout(
            title='MHW Player Count Trend: Long-term Engagement',
            xaxis_title='Date',
            yaxis_title='Average Players',
            height=400,
            hovermode='x unified'
        )
    else:
        fig4 = go.Figure()
        apply_steam_theme(fig4)
        fig4.update_layout(title='Player Count Data Not Available', height=400)

    # =================== Chart 5: Key Metrics Radar ===================
    mhw_metrics = {
        'Price Competitiveness': (50 - mhw['Price']) / 50,
        'DLC Richness': min(mhw['DLC count'] / 200, 1.0),
        'User Rating': positive_rate,
        'Market Size': min(owners_mid / 20e6, 1.0),
        'Engagement': min(mhw['Average playtime forever'] / 15000, 1.0)
    }

    similar_games = same_price_range[same_price_range['DLC count'] >= 10]
    avg_metrics = {
        'Price Competitiveness': (50 - similar_games['Price'].mean()) / 50,
        'DLC Richness': min(similar_games['DLC count'].mean() / 200, 1.0),
        'User Rating': similar_games['Positive rate'].mean(),
        'Market Size': min(similar_games['Estimated owners (mid)'].mean() / 20e6, 1.0),
        'Engagement': min(similar_games['Average playtime forever'].mean() / 15000, 1.0)
    }

    categories = list(mhw_metrics.keys())

    fig5 = go.Figure()

    fig5.add_trace(go.Scatterpolar(
        r=list(mhw_metrics.values()),
        theta=categories,
        fill='toself',
        name='Monster Hunter: World',
        line_color=COLORS['primary'],
        fillcolor='rgba(102, 192, 244, 0.3)'
    ))

    fig5.add_trace(go.Scatterpolar(
        r=list(avg_metrics.values()),
        theta=categories,
        fill='toself',
        name='Similar Games Average',
        line_color=COLORS['danger'],
        fillcolor='rgba(217, 43, 24, 0.3)'
    ))

    apply_steam_theme(fig5)
    fig5.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%'
            )
        ),
        title='MHW Performance Radar: Key Success Factors',
        height=400,
        showlegend=True
    )

    # =================== Chart 6: DLC Count vs Owners Scatter ===================
    dlc_games = same_price_range[same_price_range['DLC count'] > 0].copy()
    dlc_games = dlc_games[dlc_games['DLC count'] <= 300]

    fig6 = go.Figure()

    # Other games
    fig6.add_trace(go.Scatter(
        x=dlc_games['DLC count'],
        y=dlc_games['Estimated owners (mid)'],
        mode='markers',
        name='Other Games',
        marker=dict(
            size=8,
            color=dlc_games['Positive rate'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Rating"),
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        text=dlc_games['Name'],
        hovertemplate='<b>%{text}</b><br>DLC: %{x}<br>Owners: %{y:,.0f}<extra></extra>'
    ))

    # Highlight MHW
    fig6.add_trace(go.Scatter(
        x=[mhw['DLC count']],
        y=[owners_mid],
        mode='markers+text',
        name='Monster Hunter: World',
        marker=dict(
            size=20,
            color=COLORS['danger'],
            symbol='star',
            line=dict(width=2, color='white')
        ),
        text=['MHW'],
        textposition='top center',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>Monster Hunter: World</b><br>DLC: %{x}<br>Owners: %{y:,.0f}<extra></extra>'
    ))

    apply_steam_theme(fig6)
    fig6.update_layout(
        title='DLC Count vs Owners: Does More DLC = More Success?',
        xaxis_title='Number of DLCs',
        yaxis_title='Estimated Owners',
        yaxis_type='log',
        height=400,
        showlegend=True
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# Run the app
if __name__ == '__main__':
    print("Starting dashboard at http://127.0.0.1:8050/")
    app.run(debug=False, host='127.0.0.1', port=8050)
