import re
from PIL import Image, ImageDraw

def parse_community_data(file_path):
    """解析社区数据文件，返回社区到block坐标和平均边权的映射"""
    communities = {}
    avg_weights = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('社区'):
                # 解析完整社区行，如"社区 1: ['101', '56', ...]: ['block_6_5', ...]: 平均边权 0.7422"
                match = re.match(r'社区 (\d+):.*?: \[(.*?)\]: 平均边权 ([\d.]+)$', line)
                if match:
                    comm_id = int(match.group(1))
                    block_part = match.group(2)
                    weight = float(match.group(3))
                    # 提取所有block坐标
                    blocks = re.findall(r"'block_(\d+)_(\d+)'", block_part)
                    print(f"社区 {comm_id} 块: {blocks}, 平均边权: {weight}")
                    communities[comm_id] = [(int(row), int(col)) for row, col in blocks]
                    avg_weights[comm_id] = weight
    
    return communities, avg_weights

def visualize_communities(image_path, communities, avg_weights, output_path):
    """在图像上可视化社区"""
    # 加载图像
    img = Image.open(image_path)
    width, height = img.size
    
    # 计算每个块的大小 (假设16x16网格)
    grid_size = 16
    block_w = width // grid_size
    block_h = height // grid_size
    
    # 创建可绘制的图像
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # 根据社区数量生成区分度高的颜色
    import random
    import colorsys
    colors = []
    hue_step = 1.0 / len(communities)
    for i in range(len(communities)):
        # 使用HSV色彩空间均匀分布色相
        hue = i * hue_step
        # 固定较高的饱和度和亮度
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(r*255), int(g*255), int(b*255), 128))
    
    # 为每个社区绘制block
    for comm_id, blocks in communities.items():
        color = colors[(comm_id-1) % len(colors)]  # 社区ID从1开始
        for row, col in blocks:
            # 计算块的位置
            x1 = col * block_w
            y1 = row * block_h
            x2 = x1 + block_w
            y2 = y1 + block_h
            
            # 绘制半透明矩形
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # 在最左上角的block上绘制平均边权文本
        if blocks:
            # 找到row和col都最小的block
            min_row = min(row for row, col in blocks)
            min_col_blocks = [col for row, col in blocks if row == min_row]
            min_col = min(min_col_blocks) if min_col_blocks else 0
            
            text_x = min_col * block_w + block_w 
            text_y = min_row * block_h + block_h//2
            text = f"BLOCK{comm_id}: {avg_weights[comm_id]:.4f}"
            # 绘制文本背景
            bbox = draw.textbbox((text_x, text_y), text, anchor="ms")
            draw.rectangle([
                bbox[0],
                bbox[1],
                bbox[2] + 4,
                bbox[3] + 4
            ], fill=(255, 255, 255, 200))
            # 绘制文本
            draw.text((text_x, text_y), text, fill=(0, 0, 0), anchor="ms")
    
    # 保存结果
    img.save(output_path)
    print(f"社区可视化结果已保存到 {output_path}")

import argparse
import os

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='可视化社区检测结果')
    parser.add_argument('--community',
                       default='/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/louvain_img_community/and_interactions_s1.png.txt',
                       help='社区数据文件路径 (默认: %(default)s)')
    parser.add_argument('--image',
                       default='/mnt/data/hqdeng7/CSCIENCE/representation_bottleneck-final/datasets/ImageNet/val/n02086079/n02086079_1.jpg',
                       help='待标注的图片路径 (默认: %(default)s)')
    parser.add_argument('--output',
                       default='output/community_visualization_1.png',
                       help='输出图片路径 (默认: %(default)s)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # 解析并可视化
    communities, avg_weights = parse_community_data(args.community)
    visualize_communities(args.image, communities, avg_weights, args.output)
