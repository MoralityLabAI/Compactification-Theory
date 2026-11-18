"""Render paper figures from the byte-pinned original NOR60 artifact."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / "research/nor60_authorized_interpreter/results/nor60_first_original_hosted.zip"
EXPECTED = "c92e2c4e3bfcfd8e1371d1504c89d6812ac263ef270fc9d62e89d335f23da53c"
if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != EXPECTED:
    raise ValueError("first NOR60 archive differs from observed evidence")
with ZipFile(ARCHIVE) as archive:
    data = json.loads(archive.read("nor60_first_original_cpu.json"))

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "svg.hashsalt": "NOR60-first-evidence"})
arms = data["arms"]
labels = ["A  Original interpreter + original root",
          "B  Original interpreter + static root",
          "C  Optimized interpreter + original root",
          "D  Optimized interpreter + static root"]
ratios = [data["overall"]["source_median_ratio_of_sums_to_original"][a] for a in arms]
colors = ["#87929f", "#657d98", "#2e8791", "#124c60"]
fig, ax = plt.subplots(figsize=(9, 3.9), constrained_layout=True)
y = np.arange(4)
ax.barh(y, ratios, color=colors, height=.52)
for i, arm in enumerate(arms[1:], start=1):
    low, high = data["secondary_ten_block_bootstrap"]["descriptive_intervals"][arm]["descriptive_95pct_ten_block_interval"]
    ax.errorbar(ratios[i], i, xerr=[[ratios[i]-low], [high-ratios[i]]],
                fmt="none", ecolor="#172331", elinewidth=1.5, capsize=4)
for i, value in enumerate(ratios):
    ax.text(1.12, i, f"{value:.6f}", va="center", ha="right", fontsize=10, color="#172331")
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 1.16)
ax.axvline(1, color="#68717a", lw=1, ls=(0, (4, 3)))
ax.set_xlabel("Sum of source median verifier times / original baseline")
ax.set_title("NOR-60: complete signed verification on 60 sources", loc="left", pad=15)
ax.grid(axis="x", color="#e6e9ed", lw=.7)
ax.set_axisbelow(True)
fig.savefig(OUT / "nor60_four_arms.png", dpi=210)
fig.savefig(OUT / "nor60_four_arms.svg", metadata={"Date": None})
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 3.7), constrained_layout=True)
rows = data["source_records"]
for j, stride in enumerate((1, 4, 8)):
    group = [r for r in rows if r["source_stride"] == stride]
    values = [r["combined_over_original"] for r in group]
    offsets = np.linspace(-.22, .22, len(group))
    for offset, value, row in zip(offsets, values, group):
        color = "#b56d32" if row["program_profile"] == 4 else "#1d6a7d"
        marker = "D" if row["program_profile"] == 4 else "o"
        ax.scatter(j + offset, value, color=color, marker=marker, s=32, alpha=.86,
                   edgecolor="white", linewidth=.35, zorder=3)
    median = float(np.median(values))
    ax.plot([j-.29, j+.29], [median, median], color="#172331", lw=2.1, zorder=4)
ax.axhline(1, color="#68717a", lw=1, ls=(0, (4, 3)))
ax.set_xticks(range(3), ["Dense (s = 1)", "Support stride 4", "Support stride 8"])
ax.set_ylim(0, 1.2)
ax.set_ylabel("Combined / original source median")
ax.set_title("The improvement depends on the source and program", loc="left", pad=13)
ax.grid(axis="y", color="#e6e9ed", lw=.7)
ax.scatter([], [], color="#1d6a7d", s=32, label="Profiles P0–P3")
ax.scatter([], [], color="#b56d32", marker="D", s=32, label="P4: three reciprocal uses")
ax.plot([], [], color="#172331", lw=2.1, label="Within-stride median")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=1,
          edgecolor="none", fontsize=8.7)
fig.savefig(OUT / "nor60_source_ratios.png", dpi=210)
fig.savefig(OUT / "nor60_source_ratios.svg", metadata={"Date": None})
plt.close(fig)

with (OUT / "nor60_plotted_values.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["source_sha256", "degree", "stride", "profile", "replicate", *arms])
    for row in rows:
        writer.writerow([row["source_sha256"], row["observed_degree"], row["source_stride"],
                         row["program_profile"], row["replicate"],
                         *[row["arm_medians_ms"][a] for a in arms]])
print("Rendered two figures from the original 60-source archive.")
