"""
Plotting Utilities for Capacity-Time Dilation Experiments

This module provides standardized plotting functions for all KS experiments.
Ensures consistent styling, figure management, and export capabilities.

References:
- Hunter, "Matplotlib: A 2D Graphics Environment"
- Waskom, "seaborn: statistical data visualization"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
import os
from datetime import datetime
import warnings

# Configure matplotlib for consistent appearance
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

# Color schemes for different experiments
COLORS = {
    'KS1': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],  # Blue, orange, green, red
    'KS2': ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],  # Purple, brown, pink, gray
    'KS3': ['#bcbd22', '#17becf', '#ffbb78', '#c5b0d5'],  # Olive, cyan, light orange, light purple
    'KS4': ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']   # Light red, blue, green, orange
}

# Standard figure sizes
FIGURE_SIZES = {
    'single': (8, 6),
    'double': (12, 6),
    'quad': (15, 10),
    'tall': (8, 10),
    'wide': (15, 6)
}


def setup_figure(figsize: Union[str, Tuple[float, float]] = 'double', 
                suptitle: Optional[str] = None,
                tight_layout: bool = True) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create figure with standard formatting
    
    Args:
        figsize: Figure size ('single', 'double', 'quad', etc.) or (width, height)
        suptitle: Overall figure title
        tight_layout: Whether to apply tight_layout
        
    Returns:
        Tuple of (figure, axes)
    """
    if isinstance(figsize, str):
        figsize = FIGURE_SIZES.get(figsize, FIGURE_SIZES['double'])
    
    fig, axes = plt.subplots(figsize=figsize)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    if tight_layout:
        plt.tight_layout()
    
    return fig, axes


def setup_subplots(nrows: int, ncols: int,
                  figsize: Union[str, Tuple[float, float]] = 'quad',
                  suptitle: Optional[str] = None,
                  tight_layout: bool = True) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create subplot array with standard formatting
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size or size key
        suptitle: Overall figure title
        tight_layout: Whether to apply tight_layout
        
    Returns:
        Tuple of (figure, axes_array)
    """
    if isinstance(figsize, str):
        figsize = FIGURE_SIZES.get(figsize, FIGURE_SIZES['quad'])
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Ensure axes is always array-like for consistency
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    if tight_layout:
        plt.tight_layout()
    
    return fig, axes


def style_axis(ax: plt.Axes, 
              title: Optional[str] = None,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              grid: bool = True,
              legend: bool = False,
              xlim: Optional[Tuple[float, float]] = None,
              ylim: Optional[Tuple[float, float]] = None) -> None:
    """
    Apply standard styling to axis
    
    Args:
        ax: Matplotlib axis object
        title: Axis title
        xlabel: X-axis label
        ylabel: Y-axis label  
        grid: Whether to show grid
        legend: Whether to show legend
        xlim: X-axis limits
        ylim: Y-axis limits
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def save_figure(fig: plt.Figure, 
               filename: str,
               directory: str = 'figures',
               formats: List[str] = ['png'],
               timestamp: bool = False,
               close_after_save: bool = False) -> List[str]:
    """
    Save figure with standardized naming and format
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        directory: Output directory
        formats: List of formats to save ('png', 'pdf', 'svg', etc.)
        timestamp: Whether to add timestamp to filename
        close_after_save: Whether to close figure after saving
        
    Returns:
        List of saved file paths
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Add timestamp if requested
    if timestamp:
        timestamp_str = datetime.now().strftime("_%Y%m%d_%H%M%S")
        filename = f"{filename}{timestamp_str}"
    
    saved_paths = []
    
    for fmt in formats:
        filepath = os.path.join(directory, f"{filename}.{fmt}")
        
        try:
            fig.savefig(filepath, format=fmt, dpi=150, bbox_inches='tight')
            saved_paths.append(filepath)
            print(f"Saved: {filepath}")
        except Exception as e:
            warnings.warn(f"Failed to save {filepath}: {e}")
    
    if close_after_save:
        plt.close(fig)
    
    return saved_paths


def plot_line_comparison(x_data: Union[np.ndarray, List],
                        y_data_dict: Dict[str, Union[np.ndarray, List]],
                        title: str = "Line Comparison",
                        xlabel: str = "X",
                        ylabel: str = "Y",
                        colors: Optional[List[str]] = None,
                        styles: Optional[List[str]] = None,
                        markers: Optional[List[str]] = None,
                        figsize: Union[str, Tuple[float, float]] = 'double',
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison plot with multiple line series
    
    Args:
        x_data: X-axis data (shared across all series)
        y_data_dict: Dictionary of {label: y_data} for each series
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: List of colors for each series
        styles: List of line styles ('-', '--', ':', '-.') 
        markers: List of markers ('o', 's', '^', 'v', etc.)
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = setup_figure(figsize, title)
    
    n_series = len(y_data_dict)
    if colors is None:
        colors = COLORS['KS1'][:n_series] if n_series <= 4 else plt.cm.tab10(np.linspace(0, 1, n_series))
    if styles is None:
        styles = ['-'] * n_series
    if markers is None:
        markers = ['o', 's', '^', 'v', 'd', '*', 'p', 'h'][:n_series]
    
    for i, (label, y_data) in enumerate(y_data_dict.items()):
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        marker = markers[i % len(markers)]
        
        ax.plot(x_data, y_data, color=color, linestyle=style, marker=marker,
               label=label, markersize=4)
    
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, legend=True, grid=True)
    
    if save_path:
        save_figure(fig, save_path.replace('.png', '').replace('.pdf', ''))
    
    return fig


def plot_heatmap(data: np.ndarray,
                title: str = "Heatmap",
                xlabel: str = "X",
                ylabel: str = "Y", 
                cmap: str = 'RdBu_r',
                aspect: str = 'auto',
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                colorbar_label: Optional[str] = None,
                figsize: Union[str, Tuple[float, float]] = 'double',
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create standardized heatmap plot
    
    Args:
        data: 2D numpy array for heatmap
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name
        aspect: Aspect ratio ('auto', 'equal', or float)
        vmin, vmax: Color scale limits
        colorbar_label: Label for colorbar
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = setup_figure(figsize, title)
    
    im = ax.imshow(data.T if data.ndim == 2 else data, 
                  aspect=aspect, origin='lower', cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=False)
    
    if save_path:
        save_figure(fig, save_path.replace('.png', '').replace('.pdf', ''))
    
    return fig


def plot_scatter_with_fit(x_data: np.ndarray,
                         y_data: np.ndarray,
                         fit_type: str = 'linear',
                         title: str = "Scatter Plot with Fit",
                         xlabel: str = "X",
                         ylabel: str = "Y",
                         point_color: str = 'blue',
                         fit_color: str = 'red',
                         show_equation: bool = True,
                         figsize: Union[str, Tuple[float, float]] = 'single',
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, Dict]:
    """
    Create scatter plot with fitted line
    
    Args:
        x_data: X-axis data points
        y_data: Y-axis data points
        fit_type: Type of fit ('linear', 'quadratic', 'exponential')
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        point_color: Color for data points
        fit_color: Color for fit line
        show_equation: Whether to show fit equation on plot
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Tuple of (figure, fit_results_dict)
    """
    fig, ax = setup_figure(figsize, title)
    
    # Plot data points
    ax.scatter(x_data, y_data, color=point_color, alpha=0.7, s=40, label='Data')
    
    # Perform fit
    fit_results = {}
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    
    if fit_type == 'linear':
        coeffs = np.polyfit(x_data, y_data, 1)
        y_fit = np.polyval(coeffs, x_fit)
        equation = f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}'
        fit_results = {'slope': coeffs[0], 'intercept': coeffs[1], 'coeffs': coeffs}
        
    elif fit_type == 'quadratic':
        coeffs = np.polyfit(x_data, y_data, 2)
        y_fit = np.polyval(coeffs, x_fit)
        equation = f'y = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}'
        fit_results = {'coeffs': coeffs}
        
    elif fit_type == 'exponential':
        # Fit y = a * exp(b * x)
        try:
            log_y = np.log(y_data + 1e-12)  # Avoid log(0)
            coeffs = np.polyfit(x_data, log_y, 1)
            a, b = np.exp(coeffs[1]), coeffs[0]
            y_fit = a * np.exp(b * x_fit)
            equation = f'y = {a:.3f} * exp({b:.3f}x)'
            fit_results = {'a': a, 'b': b}
        except:
            y_fit = np.zeros_like(x_fit)
            equation = 'Fit failed'
            fit_results = {'error': 'Exponential fit failed'}
    
    # Plot fit line
    ax.plot(x_fit, y_fit, color=fit_color, linestyle='--', linewidth=2, label=f'{fit_type.title()} fit')
    
    # Add equation text
    if show_equation and 'error' not in fit_results:
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
    
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, legend=True, grid=True)
    
    if save_path:
        save_figure(fig, save_path.replace('.png', '').replace('.pdf', ''))
    
    return fig, fit_results


def plot_error_bars(x_data: np.ndarray,
                   y_data: np.ndarray,
                   y_errors: np.ndarray,
                   title: str = "Data with Error Bars",
                   xlabel: str = "X",
                   ylabel: str = "Y",
                   color: str = 'blue',
                   capsize: float = 3.0,
                   figsize: Union[str, Tuple[float, float]] = 'single',
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Create plot with error bars
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data (mean values)
        y_errors: Error bar magnitudes
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Color for points and error bars
        capsize: Size of error bar caps
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = setup_figure(figsize, title)
    
    ax.errorbar(x_data, y_data, yerr=y_errors, 
               color=color, marker='o', linestyle='-',
               capsize=capsize, capthick=1, linewidth=2, markersize=6)
    
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=True)
    
    if save_path:
        save_figure(fig, save_path.replace('.png', '').replace('.pdf', ''))
    
    return fig


def create_experiment_summary_figure(ks_results: Dict[str, bool],
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Create summary figure showing pass/fail status for all KS experiments
    
    Args:
        ks_results: Dictionary mapping experiment names to pass/fail status
        save_path: Optional save path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = setup_figure('single', "Capacity-Time Dilation Experiment Summary")
    
    experiments = list(ks_results.keys())
    results = list(ks_results.values())
    
    # Create color-coded bar chart
    colors = ['green' if result else 'red' for result in results]
    bars = ax.bar(experiments, [1]*len(experiments), color=colors, alpha=0.7)
    
    # Add pass/fail text
    for i, (exp, result) in enumerate(ks_results.items()):
        status = "PASS" if result else "FAIL"
        ax.text(i, 0.5, status, ha='center', va='center', 
               fontweight='bold', fontsize=12, color='white')
    
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Experiment Status")
    ax.set_title("Kill Switch (KS) Test Results", pad=20)
    
    # Remove y-axis ticks since they're not meaningful
    ax.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='PASS'),
                      Patch(facecolor='red', label='FAIL')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.xticks(rotation=45)
    
    if save_path:
        save_figure(fig, save_path.replace('.png', '').replace('.pdf', ''))
    
    return fig


def clear_figures() -> None:
    """Close all open matplotlib figures to free memory"""
    plt.close('all')


def set_style(style: str = 'default') -> None:
    """
    Set matplotlib style theme
    
    Args:
        style: Style name ('default', 'seaborn', 'ggplot', etc.)
    """
    try:
        plt.style.use(style)
    except:
        print(f"Warning: Style '{style}' not found, using default")
        plt.style.use('default')


# Export list for module
__all__ = [
    'setup_figure',
    'setup_subplots', 
    'style_axis',
    'save_figure',
    'plot_line_comparison',
    'plot_heatmap',
    'plot_scatter_with_fit',
    'plot_error_bars',
    'create_experiment_summary_figure',
    'clear_figures',
    'set_style',
    'COLORS',
    'FIGURE_SIZES'
]
