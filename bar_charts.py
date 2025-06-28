import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置跨平台中文字体解决方案
font_paths = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',          # WenQuanYi Micro Hei
    'SimHei'                                                   # Windows备用字体
]

# 自动检测可用字体
chinese_font = None
for path in font_paths:
    try:
        chinese_font = FontProperties(fname=path)
        break
    except:
        continue

if chinese_font is None:
    chinese_font = FontProperties()

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

jaccard_labels = ['本文方法', '梯度归因','Shapley值']
jaccard_scores = [0.79, 0.57, 0.65]

consistency_labels = ['本文方法', '梯度归因', 'Shapley值']
consistency_scores = [4.3, 3.1, 3.7]

# Jaccard系数柱状图
plt.figure(figsize=(6, 4.5), dpi=300)
bars1 = plt.bar(jaccard_labels, jaccard_scores, 
               width=0.35,
               color=['#2ecc71', '#3498db', '#9b59b6'], 
               alpha=0.8)

plt.title('Jaccard系数对比', fontproperties=chinese_font, fontsize=12, pad=15)
plt.ylim(0, 1.0)
plt.ylabel('Jaccard系数', fontproperties=chinese_font, fontsize=10)
plt.xticks(rotation=15, fontproperties=chinese_font)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}',
             ha='center', va='bottom', 
             fontsize=9, fontproperties=chinese_font)

plt.tight_layout(pad=2.5)
plt.savefig('jaccard_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

# 人工标注评分柱状图
plt.figure(figsize=(6, 4.5), dpi=300)
bars2 = plt.bar(consistency_labels, consistency_scores,
               width=0.35,
               color=['#2ecc71', '#e74c3c', '#3498db'], 
               alpha=0.8)

plt.title('人工标注一致性评分对比', fontproperties=chinese_font, fontsize=12, pad=15)
plt.ylim(0, 5.0)
plt.ylabel('评分（5分制）', fontproperties=chinese_font, fontsize=10)
plt.xticks(rotation=15, fontproperties=chinese_font)

# 添加数值标签
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}',
             ha='center', va='bottom',
             fontsize=9, fontproperties=chinese_font)

plt.tight_layout(pad=2.5)
plt.savefig('consistency_comparison.png', bbox_inches='tight', dpi=300)
plt.close()