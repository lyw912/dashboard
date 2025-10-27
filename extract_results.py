import json
import sys
from datetime import datetime

# 配置
NOTEBOOK_FILE = 'EDA.ipynb'
OUTPUT_FILE = 'notebook_outputs.txt'

def extract_notebook_outputs():
    """提取notebook中的所有文本输出并保存到文件"""

    # 读取notebook
    with open(NOTEBOOK_FILE, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 准备输出内容
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"Notebook输出提取结果")
    output_lines.append(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Notebook文件: {NOTEBOOK_FILE}")
    output_lines.append("=" * 80)
    output_lines.append("")

    # 统计信息
    total_cells = len(nb['cells'])
    cells_with_output = 0

    # 提取所有文本输出
    for i, cell in enumerate(nb['cells']):
        if 'outputs' in cell and len(cell['outputs']) > 0:
            has_text = False
            for output in cell['outputs']:
                if output.get('output_type') == 'stream':
                    has_text = True
                    break

            if has_text:
                cells_with_output += 1
                output_lines.append("")
                output_lines.append("=" * 60)
                output_lines.append(f"Cell {i}")

                # 如果有cell_id，也显示
                if 'id' in cell:
                    output_lines.append(f"Cell ID: {cell['id']}")

                output_lines.append("=" * 60)

                for output in cell['outputs']:
                    if output.get('output_type') == 'stream':
                        output_lines.append(''.join(output.get('text', [])))

    # 添加统计信息
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append(f"提取统计:")
    output_lines.append(f"  总单元格数: {total_cells}")
    output_lines.append(f"  有输出的单元格数: {cells_with_output}")
    output_lines.append("=" * 80)

    # 保存到文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    # 同时打印到控制台（前100行）
    print(f"✓ 输出已保存到: {OUTPUT_FILE}")
    print(f"✓ 提取了 {cells_with_output} 个单元格的输出")
    print(f"\n前100行预览:")
    print("-" * 80)
    for line in output_lines[:100]:
        print(line)
    if len(output_lines) > 100:
        print(f"\n... (还有 {len(output_lines) - 100} 行，请查看 {OUTPUT_FILE})")

if __name__ == '__main__':
    try:
        extract_notebook_outputs()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {NOTEBOOK_FILE}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: {NOTEBOOK_FILE} 不是有效的JSON文件")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
