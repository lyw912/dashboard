"""
案例研究分析
目标：找到符合"中低价格+多DLC"策略的典型游戏案例
"""

import pandas as pd
import numpy as np
import sys
import io

# 设置标准输出为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("案例研究：中低价格+多DLC策略")
print("=" * 80)

# 读取数据
print("\n[1] 读取数据...")
games = pd.read_csv('games_simplified.csv')

# 数据预处理
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

print(f"总游戏数: {len(games):,}")

# 筛选条件：符合"中低价格+多DLC"策略的游戏
print("\n[2] 筛选符合条件的游戏...")
print("筛选条件：")
print("  - 价格: $20-50")
print("  - DLC数量: >= 10")
print("  - 至少100条评论")
print("  - 好评率: >= 70%")
print("  - 拥有者: >= 100,000")

candidates = games[
    (games['Price'] >= 20) & (games['Price'] <= 50) &
    (games['DLC count'] >= 10) &
    (games['Total reviews'] >= 100) &
    (games['Positive rate'] >= 0.70) &
    (games['Estimated owners (mid)'] >= 100000)
].copy()

print(f"\n找到 {len(candidates)} 个符合条件的游戏")

# 按拥有者数量排序
candidates = candidates.sort_values('Estimated owners (mid)', ascending=False)

# 显示前20个候选游戏
print("\n[3] Top 20 候选游戏:")
print("-" * 120)
print(f"{'排名':<5} {'游戏名称':<40} {'价格':<8} {'DLC数':<8} {'拥有者':<15} {'好评率':<10} {'总评论':<10}")
print("-" * 120)

for idx, (_, row) in enumerate(candidates.head(20).iterrows(), 1):
    print(f"{idx:<5} {row['Name'][:38]:<40} ${row['Price']:<7.2f} {row['DLC count']:<8} "
          f"{row['Estimated owners (mid)']:>13,.0f} {row['Positive rate']:>9.1%} {row['Total reviews']:>10,}")

# 深入分析几个典型案例
print("\n" + "=" * 80)
print("[4] 深入分析 - 选择最具代表性的游戏")
print("=" * 80)

# 选择几个不同类型的代表性游戏
representative_games = []

# 1. 拥有者最多的游戏
if len(candidates) > 0:
    top_owner = candidates.iloc[0]
    representative_games.append(('拥有者最多', top_owner))

# 2. DLC数量最多的游戏（在前20中）
top_dlc = candidates.head(20).nlargest(1, 'DLC count').iloc[0] if len(candidates) >= 20 else None
if top_dlc is not None:
    representative_games.append(('DLC数量最多', top_dlc))

# 3. 性价比最高（评论数/价格比）
candidates['value_ratio'] = candidates['Total reviews'] / candidates['Price']
top_value = candidates.head(20).nlargest(1, 'value_ratio').iloc[0] if len(candidates) >= 20 else None
if top_value is not None:
    representative_games.append(('性价比最高', top_value))

# 详细显示每个代表性游戏
for category, game in representative_games:
    print(f"\n【{category}】")
    print("-" * 80)
    print(f"游戏名称: {game['Name']}")
    print(f"AppID: {game['AppID']}")
    print(f"发行日期: {game['Release date']}")
    print(f"价格: ${game['Price']:.2f}")
    print(f"DLC数量: {game['DLC count']}")
    print(f"预估拥有者: {game['Estimated owners (mid)']:,.0f}")
    print(f"好评率: {game['Positive rate']:.1%} (正面: {game['Positive']:,}, 负面: {game['Negative']:,})")
    print(f"总评论数: {game['Total reviews']:,}")
    print(f"峰值同时在线: {game['Peak CCU']:,}")
    print(f"开发商: {game['Developers']}")
    print(f"发行商: {game['Publishers']}")
    print(f"类型: {game['Genres']}")
    if not pd.isna(game['Tags']):
        tags = str(game['Tags']).split(',')[:10]
        print(f"标签: {', '.join(tags)}")
    print()

# 保存候选列表供后续使用
print("\n[5] 保存候选游戏列表...")
candidates[['AppID', 'Name', 'Price', 'DLC count', 'Estimated owners (mid)',
            'Positive rate', 'Total reviews', 'Release date', 'Developers',
            'Publishers', 'Genres']].to_csv('case_candidates.csv', index=False)
print("已保存到: case_candidates.csv")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
