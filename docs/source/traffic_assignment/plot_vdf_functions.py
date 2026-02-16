"""
Generate VDF comparison charts for documentation
"""
from aequilibrae.paths.AoN import (
    akcelik,
    bpr,
    bpr2,
    conical,
    inrets,
    delta_akcelik,
    delta_bpr,
    delta_bpr2,
    delta_conical,
    delta_inrets,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Configure matplotlib for high-quality output
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 150

# Define VDF functions following the exact implementation

array_size = 300
from_voc, to_voc = 0, 3

link_flows = np.linspace(from_voc, to_voc, array_size)


def function_apply(func, link_flows, par1, par2: float | None = None):
    size = link_flows.shape[0]
    congested_times = np.zeros(size, dtype=np.float64)
    capacity = np.ones(size, dtype=np.float64)
    fftime = np.ones(size, dtype=np.float64)
    par1s = np.ones(size, dtype=np.float64) * par1
    if par2 is not None:
        par2s = np.ones(size, dtype=np.float64) * par2
        func(congested_times, link_flows, capacity, fftime, par1s, par2s, 1)
    else:
        func(congested_times, link_flows, capacity, fftime, par1s, 1)
    return congested_times


def derivative_apply(delta_func, link_flows, par1, par2: float | None = None):
    size = link_flows.shape[0]
    derivative = np.zeros(size, dtype=np.float64)
    capacity = np.ones(size, dtype=np.float64)
    fftime = np.ones(size, dtype=np.float64)
    par1s = np.ones(size, dtype=np.float64) * par1
    if par2 is not None:
        par2s = np.ones(size, dtype=np.float64) * par2
        delta_func(derivative, link_flows, capacity, fftime, par1s, par2s, 1)
    else:
        delta_func(derivative, link_flows, capacity, fftime, par1s, 1)
    return derivative

# Create the main comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each VDF
ax.plot(link_flows, function_apply(akcelik, link_flows, 0.35, 8.0), label='Akcelik (α=0.35, τ=8.0)', linewidth=2)
ax.plot(link_flows, function_apply(bpr, link_flows, 0.15, 4.0), label='BPR (α=0.15, β=4.0)', linewidth=2)
ax.plot(link_flows, function_apply(bpr2, link_flows, 0.15, 4.0), label='BPR2 (α=0.15, β=4.0)', linewidth=2)
ax.plot(link_flows, function_apply(conical, link_flows, 1.2, 3.0), label='Conical (α=1.2, β=3.0)', linewidth=2)
ax.plot(link_flows, function_apply(inrets, link_flows, 0.9), label='INRETS (α=0.9)', linewidth=2)  # beta not used

# Add vertical line at capacity
ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Capacity (V/C = 1)')

# Formatting
ax.set_xlabel('Volume / Capacity Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Travel Time Multiplier (t / t₀)', fontsize=12, fontweight='bold')
ax.set_title('Volume Delay Functions Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(0, 3)
ax.set_ylim(0, 12)

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), '..', '_images')
os.makedirs(output_dir, exist_ok=True)

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'vdf_comparison.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'vdf_comparison.png')}")
plt.close()

# Create individual plots for each VDF with more detail
vdfs = [
    ('BPR', bpr, delta_bpr, 0.15, 4.0, 'Standard BPR function with α=0.15, β=4.0'),
    ('BPR2', bpr2, delta_bpr2, 0.15, 4.0, 'Modified BPR: β before capacity, 2β after'),
    ('Conical', conical, delta_conical, 1.2, 3.0, 'Spiess Conical with α=1.2, β=3.0'),
    ('INRETS', inrets, delta_inrets, 0.9, None, 'French INRETS with α=0.9'),
    ('Akcelik', akcelik, delta_akcelik, 0.35, 8.0, 'Akcelik function with α=0.35, τ=8.0')
]

voc_range = np.linspace(from_voc, to_voc, array_size)

for name, func, delta_func, par1, par2, description in vdfs:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    function_values = function_apply(func, voc_range, par1, par2)
    derivative_values = derivative_apply(delta_func, voc_range, par1, par2)
    
    # Left plot: Function values
    ax1.plot(voc_range, function_values, linewidth=2.5, color='#1f77b4')
    ax1.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.fill_between(voc_range, 0, function_values, alpha=0.1)
    ax1.set_xlabel('Volume / Capacity Ratio', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Travel Time Multiplier (t / t₀)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{name} VDF: Travel Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 3)
    ax1.text(1.05, ax1.get_ylim()[1] * 0.665, 'Capacity', fontsize=9, color='red', rotation=90)
    
    # Right plot: Derivative (marginal cost)
    ax2.plot(voc_range, derivative_values, linewidth=2.5, color='#ff7f0e')
    ax2.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.fill_between(voc_range, 0, derivative_values, alpha=0.1, color='#ff7f0e')
    ax2.set_xlabel('Volume / Capacity Ratio', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Marginal Travel Time (dt/dv)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{name} VDF: Marginal Cost', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 3)
    ax2.text(1.05, ax2.get_ylim()[1] * 0.665, 'Capacity', fontsize=9, color='red', rotation=90)
    
    # Add formula and description
    fig.suptitle(f'{name} - {description}', fontsize=10, y=0.98)
    
    plt.tight_layout()
    filename = f'vdf_{name.lower()}_detail.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, filename)}")
    plt.close()

# Create a comparison focused on the near-capacity region
fig, ax = plt.subplots(figsize=(10, 6))

voc_near = np.linspace(0.5, 1.5, 200)
ax.plot(voc_near, function_apply(akcelik, voc_near, 0.35, 8.0), label='Akcelik (α=0.35, τ=8.0)', linewidth=2.5)
ax.plot(voc_near, function_apply(bpr, voc_near, 0.15, 4.0), label='BPR (α=0.15, β=4.0)', linewidth=2.5)
ax.plot(voc_near, function_apply(bpr2, voc_near, 0.15, 4.0), label='BPR2 (α=0.15, β=4.0)', linewidth=2.5, linestyle='--')
ax.plot(voc_near, function_apply(conical, voc_near, 1.2, 3.0), label='Conical (α=1.2, β=3.0)', linewidth=2.5)
ax.plot(voc_near, function_apply(inrets, voc_near, 0.9), label='INRETS (α=0.9)', linewidth=2.5)

ax.axvline(x=1.0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Capacity (V/C = 1)')
ax.fill_betweenx([0, 10], 0.85, 1.15, alpha=0.1, color='yellow', label='Near-capacity region')

ax.set_xlabel('Volume / Capacity Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Travel Time Multiplier (t / t₀)', fontsize=12, fontweight='bold')
ax.set_title('VDF Comparison: Near-Capacity Behavior', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(0.5, 1.5)
ax.set_ylim(1, 4)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'vdf_near_capacity.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'vdf_near_capacity.png')}")
plt.close()

print("\nAll VDF charts generated successfully!")
