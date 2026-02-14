"""
Generate VDF comparison charts for documentation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Configure matplotlib for high-quality output
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 150

# Define VDF functions following the exact implementation


def bpr(voc, alpha=0.15, beta=4.0):
    """Bureau of Public Roads (BPR) function"""
    return 1 + alpha * (voc ** beta)


def bpr2(voc, alpha=0.15, beta=4.0):
    """Modified BPR with different behavior before and after capacity"""
    result = np.zeros_like(voc)
    mask = voc <= 1
    result[mask] = 1 + alpha * (voc[mask] ** beta)
    result[~mask] = 1 + alpha * (voc[~mask] ** (2 * beta))
    return result


def conical(voc, alpha=0.15, beta=4.0):
    """Spiess' Conical function"""
    return 2 + np.sqrt(alpha**2 * (1 - voc)**2 + beta**2) - alpha * (1 - voc) - beta


def inrets(voc, alpha=1.0):
    """French INRETS function"""
    result = np.zeros_like(voc)
    mask = voc <= 1
    # Before capacity
    result[mask] = (1.1 - alpha * voc[mask]) / (1.1 - voc[mask])
    # After capacity
    result[~mask] = ((1.1 - alpha) / 0.1) * (voc[~mask] ** 2)
    return result


def akcelik(voc, alpha=0.25, tau=0.8):
    """Akcelik function"""
    z = voc - 1.0
    return 1 + alpha * (z + np.sqrt(z**2 + tau * voc))


# Generate Volume over Capacity range
voc_range = np.linspace(0, 3, 300)

# Create the main comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each VDF
ax.plot(voc_range, bpr(voc_range), label='BPR (α=0.15, β=4.0)', linewidth=2)
ax.plot(voc_range, bpr2(voc_range), label='BPR2 (α=0.15, β=4.0)', linewidth=2, linestyle='--')
ax.plot(voc_range, conical(voc_range), label='Conical (α=0.15, β=4.0)', linewidth=2)
ax.plot(voc_range, inrets(voc_range), label='INRETS (α=1.0)', linewidth=2)
ax.plot(voc_range, akcelik(voc_range), label='Akcelik (α=0.25, τ=0.8)', linewidth=2)

# Add vertical line at capacity
ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Capacity (V/C=1)')

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
    ('BPR', lambda v: bpr(v, 0.15, 4.0), r'$t = t_0 \left(1 + \alpha \left(\frac{v}{c}\right)^\beta\right)$', 
     'Standard BPR function with α=0.15, β=4.0'),
    ('BPR2', lambda v: bpr2(v, 0.15, 4.0), 
     r'$t = t_0 \left(1 + \alpha \left(\frac{v}{c}\right)^{\beta \text{ or } 2\beta}\right)$',
     'Modified BPR: β before capacity, 2β after'),
    ('Conical', lambda v: conical(v, 0.15, 4.0),
     r'$t = t_0 \left(2 + \sqrt{\alpha^2\left(1-\frac{v}{c}\right)^2 + \beta^2} - \alpha\left(1-\frac{v}{c}\right) - \beta\right)$',
     'Spiess Conical with α=0.15, β=4.0'),
    ('INRETS', lambda v: inrets(v, 1.0),
     r'Before capacity: $t = t_0 \frac{1.1 - \alpha\frac{v}{c}}{1.1 - \frac{v}{c}}$',
     'French INRETS with α=1.0'),
    ('Akcelik', lambda v: akcelik(v, 0.25, 0.8),
     r'$t = t_0 + \alpha\left(z + \sqrt{z^2 + \frac{\tau v}{c}}\right), z = \frac{v}{c} - 1$',
     'Akcelik function with α=0.25, τ=0.8')
]

for name, func, formula, description in vdfs:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left plot: Function values
    ax1.plot(voc_range, func(voc_range), linewidth=2.5, color='#1f77b4')
    ax1.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.fill_between(voc_range, 0, func(voc_range), alpha=0.1)
    ax1.set_xlabel('Volume / Capacity Ratio', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Travel Time Multiplier (t / t₀)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{name} VDF: Travel Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 3)
    ax1.text(1.05, ax1.get_ylim()[1] * 0.95, 'Capacity', fontsize=9, color='red', rotation=90)
    
    # Right plot: Derivative (marginal cost)
    dx = 0.001
    derivative = np.gradient(func(voc_range), dx)
    ax2.plot(voc_range, derivative, linewidth=2.5, color='#ff7f0e')
    ax2.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.fill_between(voc_range, 0, derivative, alpha=0.1, color='#ff7f0e')
    ax2.set_xlabel('Volume / Capacity Ratio', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Marginal Travel Time (dt/dv)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{name} VDF: Marginal Cost', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 3)
    ax2.text(1.05, ax2.get_ylim()[1] * 0.95, 'Capacity', fontsize=9, color='red', rotation=90)
    
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
ax.plot(voc_near, bpr(voc_near), label='BPR', linewidth=2.5)
ax.plot(voc_near, bpr2(voc_near), label='BPR2', linewidth=2.5, linestyle='--')
ax.plot(voc_near, conical(voc_near), label='Conical', linewidth=2.5)
ax.plot(voc_near, inrets(voc_near), label='INRETS', linewidth=2.5)
ax.plot(voc_near, akcelik(voc_near), label='Akcelik', linewidth=2.5)

ax.axvline(x=1.0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Capacity')
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
