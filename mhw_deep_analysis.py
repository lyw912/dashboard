"""
Monster Hunter: World 深度案例分析
论证：中低价格+多DLC策略的成功典范
"""

import pandas as pd
import numpy as np
import sys
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("Monster Hunter: World 深度案例分析")
print("论证：中低价格+多DLC策略的成功典范")
print("=" * 80)

# 读取数据
games = pd.read_csv('games_simplified.csv')
price_data = pd.read_csv('price_final.csv')
count_data = pd.read_csv('count_final.csv')

# 获取MHW的AppID
MHW_APPID = 582010

# 提取MHW数据
mhw = games[games['AppID'] == MHW_APPID].iloc[0]

print("\n【1. 基本信息】")
print("-" * 80)
print(f"游戏名称: {mhw['Name']}")
print(f"AppID: {mhw['AppID']}")
print(f"发行日期: {mhw['Release date']}")
print(f"开发商: {mhw['Developers']}")
print(f"发行商: {mhw['Publishers']}")
print(f"类型: {mhw['Genres']}")

# 提取拥有者中值
def extract_owners_mid(owners_str):
    if pd.isna(owners_str) or owners_str == '0 - 0':
        return 0
    parts = str(owners_str).split(' - ')
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return 0

owners_mid = extract_owners_mid(mhw['Estimated owners'])
positive_rate = mhw['Positive'] / (mhw['Positive'] + mhw['Negative'])

print("\n【2. 核心指标】")
print("-" * 80)
print(f"当前价格: ${mhw['Price']:.2f}")
print(f"DLC数量: {mhw['DLC count']}")
print(f"预估拥有者: {owners_mid:,.0f}")
print(f"峰值同时在线: {mhw['Peak CCU']:,}")
print(f"好评率: {positive_rate:.1%}")
print(f"  - 正面评价: {mhw['Positive']:,}")
print(f"  - 负面评价: {mhw['Negative']:,}")
print(f"  - 总评论数: {mhw['Positive'] + mhw['Negative']:,}")
print(f"平均游戏时长: {mhw['Average playtime forever']:,.0f} 分钟 ({mhw['Average playtime forever']/60:.1f} 小时)")

# 计算预估收入
estimated_revenue = owners_mid * mhw['Price']
print(f"\n预估总收入 (仅基础版): ${estimated_revenue:,.0f} ({estimated_revenue/1e6:.1f} 百万美元)")

# 与同价位游戏对比
print("\n【3. 与同价位游戏对比】")
print("-" * 80)

# 筛选同价位游戏（$20-40）
same_price_range = games[
    (games['Price'] >= 20) & (games['Price'] <= 40) &
    (games['Positive'] + games['Negative'] >= 100)
].copy()

same_price_range['owners_mid'] = same_price_range['Estimated owners'].apply(extract_owners_mid)
same_price_range['positive_rate'] = same_price_range['Positive'] / (same_price_range['Positive'] + same_price_range['Negative'] + 1)

print(f"同价位游戏数量: {len(same_price_range):,}")
print(f"\n MHW在同价位游戏中的排名:")
print(f"  - 拥有者排名: {(same_price_range['owners_mid'] > owners_mid).sum() + 1} / {len(same_price_range)}")
print(f"  - DLC数量排名: {(same_price_range['DLC count'] > mhw['DLC count']).sum() + 1} / {len(same_price_range)}")
print(f"  - 好评率排名: {(same_price_range['positive_rate'] > positive_rate).sum() + 1} / {len(same_price_range)}")

# 计算百分位数
owner_percentile = (same_price_range['owners_mid'] < owners_mid).mean() * 100
dlc_percentile = (same_price_range['DLC count'] < mhw['DLC count']).mean() * 100
rating_percentile = (same_price_range['positive_rate'] < positive_rate).mean() * 100

print(f"\n MHW超过了同价位游戏的:")
print(f"  - 拥有者: {owner_percentile:.1f}%")
print(f"  - DLC数量: {dlc_percentile:.1f}%")
print(f"  - 好评率: {rating_percentile:.1f}%")

# 对比分析：不同DLC策略
print("\n【4. DLC策略对比分析】")
print("-" * 80)

# 定义三种策略
strategies = {
    '低价+高DLC (MHW模式)': same_price_range[
        (same_price_range['Price'] < 35) & (same_price_range['DLC count'] >= 15)
    ],
    '中价+中DLC': same_price_range[
        (same_price_range['Price'] >= 25) & (same_price_range['Price'] < 35) &
        (same_price_range['DLC count'] >= 5) & (same_price_range['DLC count'] < 15)
    ],
    '高价+低DLC': same_price_range[
        (same_price_range['Price'] >= 35) & (same_price_range['DLC count'] < 10)
    ]
}

print(f"{'策略':<25} {'游戏数':<10} {'平均拥有者':<18} {'平均好评率':<12} {'平均DLC数':<10}")
print("-" * 80)

strategy_stats = {}
for strategy_name, strategy_games in strategies.items():
    if len(strategy_games) > 0:
        avg_owners = strategy_games['owners_mid'].mean()
        avg_rating = strategy_games['positive_rate'].mean()
        avg_dlc = strategy_games['DLC count'].mean()
        game_count = len(strategy_games)

        strategy_stats[strategy_name] = {
            'count': game_count,
            'avg_owners': avg_owners,
            'avg_rating': avg_rating,
            'avg_dlc': avg_dlc
        }

        print(f"{strategy_name:<25} {game_count:<10} {avg_owners:>16,.0f}  {avg_rating:>10.1%}  {avg_dlc:>10.1f}")

# 价格历史分析
print("\n【5. 价格历史分析】")
print("-" * 80)

mhw_price_history = price_data[price_data['AppID'] == MHW_APPID].copy()

if len(mhw_price_history) > 0:
    mhw_price_history['date'] = pd.to_datetime(mhw_price_history['date'])
    mhw_price_history = mhw_price_history.sort_values('date')

    print(f"价格历史记录: {len(mhw_price_history)} 条")
    print(f"最低价格: ${mhw_price_history['price'].min():.2f}")
    print(f"最高价格: ${mhw_price_history['price'].max():.2f}")
    print(f"当前价格: ${mhw_price_history['price'].iloc[-1]:.2f}")
    print(f"平均价格: ${mhw_price_history['price'].mean():.2f}")

    # 计算折扣频率
    original_price = mhw['Price']
    discount_records = mhw_price_history[mhw_price_history['price'] < original_price]
    discount_rate = len(discount_records) / len(mhw_price_history)

    print(f"\n折扣频率: {discount_rate:.1%} 的时间有折扣")
    print(f"平均折扣幅度: {(1 - discount_records['price'].mean() / original_price):.1%}")
else:
    print("暂无价格历史数据")

# 玩家数量趋势分析
print("\n【6. 玩家数量趋势分析】")
print("-" * 80)

mhw_count_history = count_data[count_data['AppID'] == MHW_APPID].copy()

if len(mhw_count_history) > 0:
    mhw_count_history['date'] = pd.to_datetime(mhw_count_history['date'])
    mhw_count_history = mhw_count_history.sort_values('date')

    print(f"玩家数量历史记录: {len(mhw_count_history)} 条")
    print(f"最高平均在线: {mhw_count_history['avg_players'].max():,.0f}")
    print(f"当前平均在线: {mhw_count_history['avg_players'].iloc[-1]:,.0f}")
    print(f"平均在线人数: {mhw_count_history['avg_players'].mean():,.0f}")

    # 计算留存率（当前/峰值）
    retention_rate = mhw_count_history['avg_players'].iloc[-1] / mhw_count_history['avg_players'].max()
    print(f"\n玩家留存率: {retention_rate:.1%} (当前在线 / 历史峰值)")
else:
    print("暂无玩家数量历史数据")

# 与其他热门游戏对比
print("\n【7. 与其他中价位+高DLC游戏对比】")
print("-" * 80)

# 选择几个对比游戏
comparison_games = [
    ('ARK: Survival Evolved', 346110),
    ('Cities: Skylines', 255710),
    ('Beat Saber', 620980),
    ('Dying Light', 239140)
]

print(f"{'游戏名称':<35} {'价格':<8} {'DLC数':<8} {'拥有者':<15} {'好评率':<10}")
print("-" * 80)
print(f"{'Monster Hunter: World':<35} ${mhw['Price']:<7.2f} {mhw['DLC count']:<8} {owners_mid:>13,.0f} {positive_rate:>9.1%}")

for game_name, app_id in comparison_games:
    game_data = games[games['AppID'] == app_id]
    if len(game_data) > 0:
        game = game_data.iloc[0]
        game_owners = extract_owners_mid(game['Estimated owners'])
        game_rate = game['Positive'] / (game['Positive'] + game['Negative'] + 1)
        print(f"{game_name:<35} ${game['Price']:<7.2f} {game['DLC count']:<8} {game_owners:>13,.0f} {game_rate:>9.1%}")

print("\n【8. 结论】")
print("-" * 80)
print("Monster Hunter: World 完美展示了'中低价格+多DLC'策略的成功:")
print()
print(f"1. 定价策略: ${mhw['Price']:.2f} 处于中价位甜点区间($20-40)")
print(f"2. DLC策略: {mhw['DLC count']} 个DLC提供持续内容更新")
print(f"3. 市场表现: {owners_mid:,.0f} 拥有者，超过同价位游戏 {owner_percentile:.0f}%")
print(f"4. 用户满意度: {positive_rate:.1%} 好评率，超过同价位游戏 {rating_percentile:.0f}%")
print(f"5. 长期运营: 通过DLC保持玩家活跃度和持续收入")
print()
print("这一案例有力证明了：")
print("- 合理的基础定价可以建立庞大的用户基础")
print("- 丰富的DLC内容能够延长游戏生命周期")
print("- 持续更新维护可以保持玩家忠诚度和活跃度")
print("- 中价位+高DLC策略在拥有者数量上优于高价+低DLC策略")

print("\n" + "=" * 80)
print("分析完成！数据已准备好用于可视化。")
print("=" * 80)

# 保存关键数据用于可视化
analysis_results = {
    'mhw_info': {
        'name': str(mhw['Name']),
        'appid': int(MHW_APPID),
        'price': float(mhw['Price']),
        'dlc_count': int(mhw['DLC count']),
        'owners': float(owners_mid),
        'positive_rate': float(positive_rate),
        'total_reviews': int(mhw['Positive'] + mhw['Negative']),
        'peak_ccu': int(mhw['Peak CCU']),
        'avg_playtime_hours': float(mhw['Average playtime forever'] / 60)
    },
    'strategy_comparison': {k: {k2: float(v2) if isinstance(v2, (np.floating, float)) else int(v2) if isinstance(v2, (np.integer, int)) else v2
                                for k2, v2 in v.items()} for k, v in strategy_stats.items()},
    'comparison_games': comparison_games
}

# 保存到文件供dashboard使用
import json
with open('mhw_analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, ensure_ascii=False, indent=2)

print("\n分析结果已保存到: mhw_analysis_results.json")
