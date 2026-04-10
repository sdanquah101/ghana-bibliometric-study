import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path("analysis_results")

# Setup figure
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

# Colors
box_color = "#EBF5FB"  # Light blue
border_color = "#2874A6" # Dark blue
text_color = "#154360"
arrow_color = "#5D6D7E"
exclude_color = "#FDEDEC" # Light red
exclude_border = "#C0392B" # Dark red

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor):
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', color=text_color, fontsize=11, linespacing=1.6)

def draw_arrow(ax, x, y, dx, dy):
    ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
                zorder=1)

def draw_elbow_arrow(ax, start_x, start_y, end_x, end_y):
    # Down then right
    ax.plot([start_x, start_x], [start_y, end_y], color=arrow_color, lw=2, zorder=1)
    ax.annotate("", xy=(end_x, end_y), xytext=(start_x, end_y),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
                zorder=1)

# Coordinates and dimensions
bw = 4.0   # box width
bh = 1.0   # box height
cx = 3.5   # center x
ex = 8.5   # excluded box x

# Nodes data
nodes = [
    {"y": 8.5, "text": "Records identified through OpenAlex\n(Ghanaian institutions, Biomed/Eng fields)\n$\\mathbf{n = 46,945}$"},
    {"y": 6.5, "text": "Publications within study period\n(Years 2000-2025)\n$\\mathbf{n = 41,788}$"},
    {"y": 4.5, "text": "International collaborations\n(≥1 non-Ghanaian affiliation)\n$\\mathbf{n = 25,949}$"},
    {"y": 2.5, "text": "Final study dataset\n(≥2 authors)\n$\\mathbf{n = 24,768}$"}
]

exclude_nodes = [
    {"y": 7.5, "text": "Excluded: Outside date range\n$\\mathbf{n = 5,157}$"},
    {"y": 5.5, "text": "Excluded: Domestic-only works\n$\\mathbf{n = 15,839}$"},
    {"y": 3.5, "text": "Excluded: Single-author works\n$\\mathbf{n = 1,181}$"}
]

# Draw main boxes
for n in nodes:
    draw_box(ax, cx - bw/2, n["y"], bw, bh, n["text"], box_color, border_color)

# Draw excluded boxes
for n in exclude_nodes:
    draw_box(ax, ex - bw/2, n["y"], bw, bh, n["text"], exclude_color, exclude_border)

# Draw main flow arrows
for i in range(len(nodes)-1):
    top_y = nodes[i]["y"]
    bot_y = nodes[i+1]["y"] + bh
    draw_arrow(ax, cx, top_y, 0, bot_y - top_y)

# Draw exclusion elbow arrows
for i in range(len(exclude_nodes)):
    top_y = nodes[i]["y"]
    exc_y = exclude_nodes[i]["y"] + bh/2
    exc_x = ex - bw/2
    draw_elbow_arrow(ax, cx, top_y - 0.2, exc_x, exc_y)

ax.set_xlim(0, 11)
ax.set_ylim(1, 10)
fig.tight_layout()

file_path = OUTPUT_DIR / "chart_00_prisma_flow.png"
fig.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated {file_path}")
