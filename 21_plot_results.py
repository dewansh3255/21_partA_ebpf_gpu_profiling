#!/usr/bin/env python3
"""
21_plot_results.py
Group 21 - GRS Project Part A

Matplotlib-based visualization script for comparing container vs native
profiling results. Generates publication-quality plots for the report.

NOTE: As per submission guidelines, values are hardcoded in the script.
      CSV files are NOT used as input. Raw CSVs are submitted separately.

Usage:
    python3 21_plot_results.py

Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    """Configure publication-quality plot styling."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_syscall_comparison():
    """
    Figure 1: Syscall Count Comparison — Native vs Containerized
    Hardcoded values from initial profiling run.
    """
    # Syscall names
    syscalls = ['futex', 'ioctl', 'read', 'write', 'mmap',
                'clone', 'openat', 'close', 'epoll_wait', 'recvmsg']

    # Counts from native run (placeholder values from initial tests)
    native_counts = [45230, 32100, 28400, 21300, 15600,
                     8900, 7200, 6800, 5400, 3200]

    # Counts from containerized run (typically higher due to cgroup/ns overhead)
    container_counts = [52800, 38700, 31200, 24100, 18900,
                        11200, 9800, 8500, 7100, 4800]

    x = np.arange(len(syscalls))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, native_counts, width, label='Native',
                   color='#2196F3', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, container_counts, width, label='Containerized',
                   color='#FF5722', alpha=0.85, edgecolor='white')

    ax.set_xlabel('System Call')
    ax.set_ylabel('Count')
    ax.set_title('Syscall Frequency: Native vs Containerized (Expected / Simulated Data)')
    ax.set_xticks(x)
    ax.set_xticklabels(syscalls, rotation=45, ha='right')
    ax.legend()

    # Add percentage overhead labels
    for i, (n, c) in enumerate(zip(native_counts, container_counts)):
        overhead = ((c - n) / n) * 100
        ax.annotate(f'+{overhead:.0f}%',
                    xy=(x[i] + width/2, c),
                    ha='center', va='bottom',
                    fontsize=8, color='#D32F2F', fontweight='bold')

    plt.tight_layout()
    plt.savefig('21_plot_syscall_comparison.png', bbox_inches='tight')
    print("[Plot] Saved: 21_plot_syscall_comparison.png")
    plt.close()


def plot_scheduling_latency():
    """
    Figure 2: Scheduling Latency CDF — Native vs Containerized
    Shows the cumulative distribution of scheduler run-queue wait times.
    """
    # Simulated latency data (microseconds) — sorted
    np.random.seed(21)  # Reproducible

    native_latencies = sorted(np.concatenate([
        np.random.exponential(5, 800),     # Most are fast
        np.random.exponential(20, 150),    # Some moderate
        np.random.exponential(100, 50),    # Few slow
    ]))

    container_latencies = sorted(np.concatenate([
        np.random.exponential(8, 700),     # Shifted higher
        np.random.exponential(30, 200),    # More moderate
        np.random.exponential(150, 100),   # More slow (cgroup overhead)
    ]))

    fig, ax = plt.subplots(figsize=(10, 6))

    # CDF
    native_cdf = np.arange(1, len(native_latencies) + 1) / len(native_latencies)
    container_cdf = np.arange(1, len(container_latencies) + 1) / len(container_latencies)

    ax.plot(native_latencies, native_cdf, label='Native',
            color='#2196F3', linewidth=2)
    ax.plot(container_latencies, container_cdf, label='Containerized',
            color='#FF5722', linewidth=2)

    # Mark P50, P95, P99
    for latencies, cdf, color, name in [
        (native_latencies, native_cdf, '#2196F3', 'Native'),
        (container_latencies, container_cdf, '#FF5722', 'Container')
    ]:
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        ax.axhline(y=0.50, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel('Scheduling Latency (μs)')
    ax.set_ylabel('CDF')
    ax.set_title('CPU Scheduling Latency Distribution (Expected / Simulated Data)')
    ax.legend()
    ax.set_xscale('log')

    # Add text annotations for percentiles
    ax.text(0.02, 0.50, 'P50', transform=ax.get_yaxis_transform(),
            fontsize=9, color='gray', va='bottom')
    ax.text(0.02, 0.95, 'P95', transform=ax.get_yaxis_transform(),
            fontsize=9, color='gray', va='bottom')
    ax.text(0.02, 0.99, 'P99', transform=ax.get_yaxis_transform(),
            fontsize=9, color='gray', va='bottom')

    plt.tight_layout()
    plt.savefig('21_plot_scheduling_latency.png', bbox_inches='tight')
    print("[Plot] Saved: 21_plot_scheduling_latency.png")
    plt.close()


def plot_network_latency():
    """
    Figure 3: Network Processing Latency — Native vs Containerized
    Box plot showing TCP send/recv latency distributions.
    """
    np.random.seed(42)

    # Network latency data (microseconds)
    data = {
        'Native\ntcp_send': np.random.lognormal(mean=2.0, sigma=0.5, size=200),
        'Container\ntcp_send': np.random.lognormal(mean=2.5, sigma=0.6, size=200),
        'Native\ntcp_recv': np.random.lognormal(mean=1.8, sigma=0.4, size=200),
        'Container\ntcp_recv': np.random.lognormal(mean=2.3, sigma=0.5, size=200),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data.values(), labels=data.keys(), patch_artist=True,
                    showfliers=False, notch=True)

    colors = ['#2196F3', '#FF5722', '#2196F3', '#FF5722']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Latency (μs)')
    ax.set_title('Network Stack Latency: Native vs Containerized (Expected / Simulated Data)')

    # Add median annotations
    for i, (name, vals) in enumerate(data.items()):
        median = np.median(vals)
        ax.annotate(f'med={median:.1f}μs',
                    xy=(i + 1, median), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('21_plot_network_latency.png', bbox_inches='tight')
    print("[Plot] Saved: 21_plot_network_latency.png")
    plt.close()


def plot_gpu_utilization():
    """
    Figure 4: GPU Utilization Timeline — Native vs Containerized
    Shows GPU utilization over time during training.
    """
    np.random.seed(21)

    # Time axis (seconds)
    time_axis = np.arange(0, 60, 0.5)  # 60s at 0.5s intervals

    # GPU utilization (%)
    # Native: generally higher, more stable
    native_util = np.clip(
        75 + 15 * np.sin(time_axis * 0.3) + np.random.normal(0, 5, len(time_axis)),
        0, 100
    )

    # Container: slightly lower due to overhead, more variance
    container_util = np.clip(
        68 + 15 * np.sin(time_axis * 0.3) + np.random.normal(0, 8, len(time_axis)),
        0, 100
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time_axis, native_util, color='#2196F3', alpha=0.7, linewidth=1)
    ax1.fill_between(time_axis, native_util, alpha=0.2, color='#2196F3')
    ax1.set_ylabel('GPU Utilization (%)')
    ax1.set_title('GPU Utilization Timeline — Native (Expected / Simulated Data)')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=np.mean(native_util), color='#2196F3', linestyle='--',
                alpha=0.5, label=f'Mean: {np.mean(native_util):.1f}%')
    ax1.legend()

    ax2.plot(time_axis, container_util, color='#FF5722', alpha=0.7, linewidth=1)
    ax2.fill_between(time_axis, container_util, alpha=0.2, color='#FF5722')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('GPU Utilization (%)')
    ax2.set_title('GPU Utilization Timeline — Containerized (Expected / Simulated Data)')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=np.mean(container_util), color='#FF5722', linestyle='--',
                alpha=0.5, label=f'Mean: {np.mean(container_util):.1f}%')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('21_plot_gpu_utilization.png', bbox_inches='tight')
    print("[Plot] Saved: 21_plot_gpu_utilization.png")
    plt.close()


def plot_overhead_summary():
    """
    Figure 5: Overall Overhead Summary — Container vs Native
    Bar chart showing percentage overhead across all metrics.
    """
    metrics = [
        'Total Syscalls',
        'Sched Latency\n(P95)',
        'TCP Send\nLatency',
        'TCP Recv\nLatency',
        'GPU Util\nDrop',
        'Training\nTime',
    ]

    # Percentage overhead of container vs native
    overheads = [16.7, 22.4, 35.2, 28.1, 9.3, 12.5]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#FF5722' if v > 20 else '#FF9800' if v > 10 else '#4CAF50'
              for v in overheads]

    bars = ax.bar(metrics, overheads, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    ax.set_ylabel('Overhead (%)')
    ax.set_title('Overhead Summary: Container vs Native (Expected / Simulated Data)')

    # Add value labels
    for bar, val in zip(bars, overheads):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', alpha=0.85, label='Low (<10%)'),
        Patch(facecolor='#FF9800', alpha=0.85, label='Moderate (10-20%)'),
        Patch(facecolor='#FF5722', alpha=0.85, label='High (>20%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_ylim(0, max(overheads) * 1.25)
    plt.tight_layout()
    plt.savefig('21_plot_overhead_summary.png', bbox_inches='tight')
    print("[Plot] Saved: 21_plot_overhead_summary.png")
    plt.close()


def main():
    setup_style()

    print("=" * 50)
    print("Generating plots for Group 21 report...")
    print("=" * 50)

    plot_syscall_comparison()
    plot_scheduling_latency()
    plot_network_latency()
    plot_gpu_utilization()
    plot_overhead_summary()

    print("\n" + "=" * 50)
    print("All plots generated successfully!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - 21_plot_syscall_comparison.png")
    print("  - 21_plot_scheduling_latency.png")
    print("  - 21_plot_network_latency.png")
    print("  - 21_plot_gpu_utilization.png")
    print("  - 21_plot_overhead_summary.png")


if __name__ == "__main__":
    main()
