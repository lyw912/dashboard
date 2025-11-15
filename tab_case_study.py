"""
Case Study Tab - Monster Hunter: World
用于测试可视化，确认后整合到dashboard_app.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 读取数据
print("Loading data...")
games = pd.read_csv('games_simplified.csv')
price_data = pd.read_csv('price_final.csv')
count_data = pd.read_csv('count_final.csv')

# MHW AppID
MHW_APPID = 582010

# 数据处理函数
def extract_owners_mid(owners_str):
    if pd.isna(owners_str) or owners_str == '0 - 0':
        return 0
    parts = str(owners_str).split(' - ')
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return 0

games['Estimated owners (mid)'] = games['Estimated owners'].apply(extract_owners_mid)
games['Positive rate'] = games['Positive'] / (games['Positive'] + games['Negative'] + 1)
games['Total reviews'] = games['Positive'] + games['Negative']

# 提取MHW数据
mhw = games[games['AppID'] == MHW_APPID].iloc[0]
owners_mid = extract_owners_mid(mhw['Estimated owners'])
positive_rate = mhw['Positive'] / (mhw['Positive'] + mhw['Negative'])

# Steam颜色主题
COLORS = {
    'primary': '#66c0f4',
    'success': "#FFFF00",
    'warning': "#12f33b",
    'danger': "#d92b18",
    'info': "#8266f4",
    'dark': '#171a21',
    'light': "#ffffff",
    'bg_dark': '#1b2838',
    'bg_gradient': '#2a475e'
}

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

print("\n=== Creating Visualizations ===\n")

# =================== 图1: DLC策略对比 ===================
print("1. DLC Strategy Comparison...")

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

print(f"   Strategy comparison: {len(strategy_df)} strategies")

# =================== 图2: MHW vs 竞品对比 ===================
print("2. MHW vs Competitors...")

comparison_games = [
    ('Monster Hunter: World', 582010),
    ('ARK: Survival Evolved', 346110),
    ('Cities: Skylines', 255710),
    ('Beat Saber', 620980),
    ('Dying Light', 239140)
]

comparison_data = []
for game_name, app_id in comparison_games:
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

# 添加拥有者数量柱状图
fig2.add_trace(go.Bar(
    name='Owners (Millions)',
    x=comparison_df['Game'],
    y=comparison_df['Owners'] / 1e6,
    marker_color=COLORS['primary'],
    yaxis='y',
    text=[f"{v/1e6:.1f}M" for v in comparison_df['Owners']],
    textposition='outside'
))

# 添加好评率折线图
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

print(f"   Compared {len(comparison_df)} games")

# =================== 图3: 价格历史趋势 ===================
print("3. Price History Trend...")

mhw_price_history = price_data[price_data['AppID'] == MHW_APPID].copy()
mhw_price_history['date'] = pd.to_datetime(mhw_price_history['date'])
mhw_price_history = mhw_price_history.sort_values('date')

if len(mhw_price_history) > 0:
    fig3 = go.Figure()

    # 价格折线
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

    # 原价参考线
    fig3.add_hline(
        y=mhw['Price'],
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text=f"Original Price: ${mhw['Price']:.2f}",
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

    print(f"   Price history: {len(mhw_price_history)} records")
else:
    fig3 = go.Figure()
    apply_steam_theme(fig3)
    fig3.update_layout(title='Price History Not Available', height=400)
    print("   No price history data")

# =================== 图4: 玩家数量趋势 ===================
print("4. Player Count Trend...")

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

    # 标注峰值
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

    print(f"   Player count: {len(mhw_count_history)} records")
else:
    fig4 = go.Figure()
    apply_steam_theme(fig4)
    fig4.update_layout(title='Player Count Data Not Available', height=400)
    print("   No player count data")

# =================== 图5: 关键指标雷达图 ===================
print("5. Key Metrics Radar...")

# MHW vs 平均值对比
mhw_metrics = {
    'Price Competitiveness': (50 - mhw['Price']) / 50,  # 价格越低越有竞争力
    'DLC Richness': min(mhw['DLC count'] / 200, 1.0),  # DLC丰富度
    'User Rating': positive_rate,
    'Market Size': min(owners_mid / 20e6, 1.0),  # 市场规模
    'Engagement': min(mhw['Average playtime forever'] / 15000, 1.0)  # 参与度
}

# 同类游戏平均值
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

print("   Radar chart created")

# =================== 图6: DLC数量vs拥有者散点图 ===================
print("6. DLC Count vs Owners Scatter...")

# 筛选有DLC的游戏
dlc_games = same_price_range[same_price_range['DLC count'] > 0].copy()
dlc_games = dlc_games[dlc_games['DLC count'] <= 300]  # 去除极端值

fig6 = go.Figure()

# 其他游戏散点
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

# 突出显示MHW
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

print("   Scatter plot created")

print("\n=== All visualizations created successfully! ===")
print("\nReady to integrate into dashboard_app.py")

# 显示关键统计
print("\n=== Key Statistics ===")
print(f"MHW Owners: {owners_mid:,.0f}")
print(f"MHW DLC Count: {mhw['DLC count']}")
print(f"MHW Rating: {positive_rate:.1%}")
print(f"Strategy comparison games: {sum(len(v) for v in strategies.values())}")
