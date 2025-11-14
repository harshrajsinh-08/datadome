import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import os

def visualize_image_cleaning_results(original_stats, cleaned_stats, cleaning_stats, save_path):
    """Create comprehensive visualization for image dataset cleaning results"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    # Main title
    fig.suptitle('DataDome Image Dataset Cleaning Results', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Before/After Count Comparison
    categories = ['Initial\nDataset', 'Cleaned\nDataset']
    counts = [cleaning_stats['initial_count'], cleaning_stats['final_count']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = axes[0].bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].set_title('Dataset Size Comparison', fontsize=14, pad=20)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Cleaning Steps Breakdown
    cleaning_categories = [
        'Corrupted', 'Low Res', 'High Res', 'Blurry', 
        'Brightness', 'Low Contrast', 'Duplicates'
    ]
    cleaning_counts = [
        cleaning_stats['corrupted_removed'],
        cleaning_stats['low_resolution_removed'],
        cleaning_stats['high_resolution_removed'],
        cleaning_stats['blurry_removed'],
        cleaning_stats['brightness_issues_removed'],
        cleaning_stats['low_contrast_removed'],
        cleaning_stats['duplicates_removed']
    ]
    
    bars = axes[1].bar(cleaning_categories, cleaning_counts, color='#FF6B6B', alpha=0.7)
    axes[1].set_title('Images Removed by Category', fontsize=14, pad=20)
    axes[1].set_ylabel('Number of Images Removed', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, cleaning_counts):
        if count > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cleaning_counts)*0.01, 
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Quality Metrics Comparison (Before vs After)
    if 'cleaned_df' in locals() or hasattr(original_stats, 'cleaned_df'):
        # This would need the actual cleaned dataframe
        # For now, show original distribution
        pass
    
    # 3. Resolution Distribution (Original)
    if len(original_stats) > 0:
        axes[2].hist(original_stats['resolution'], bins=30, alpha=0.7, color='#FF6B6B', 
                    label='Original', edgecolor='black', linewidth=0.5)
        axes[2].set_title('Resolution Distribution (Original)', fontsize=14, pad=20)
        axes[2].set_xlabel('Resolution (pixels)', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].grid(axis='y', alpha=0.3)
    
    # 4. Blur Score Distribution
    if len(original_stats) > 0:
        axes[3].hist(original_stats['blur_score'], bins=30, alpha=0.7, color='#4ECDC4',
                    edgecolor='black', linewidth=0.5)
        axes[3].set_title('Blur Score Distribution (Original)', fontsize=14, pad=20)
        axes[3].set_xlabel('Blur Score (higher = sharper)', fontsize=12)
        axes[3].set_ylabel('Frequency', fontsize=12)
        axes[3].grid(axis='y', alpha=0.3)
    
    # 5. Brightness Distribution
    if len(original_stats) > 0:
        axes[4].hist(original_stats['brightness'], bins=30, alpha=0.7, color='#FFD93D',
                    edgecolor='black', linewidth=0.5)
        axes[4].set_title('Brightness Distribution (Original)', fontsize=14, pad=20)
        axes[4].set_xlabel('Average Brightness', fontsize=12)
        axes[4].set_ylabel('Frequency', fontsize=12)
        axes[4].grid(axis='y', alpha=0.3)
    
    # 6. File Size Distribution
    if len(original_stats) > 0:
        # Convert to MB for better readability
        file_sizes_mb = original_stats['file_size'] / (1024 * 1024)
        axes[5].hist(file_sizes_mb, bins=30, alpha=0.7, color='#9B59B6',
                    edgecolor='black', linewidth=0.5)
        axes[5].set_title('File Size Distribution (Original)', fontsize=14, pad=20)
        axes[5].set_xlabel('File Size (MB)', fontsize=12)
        axes[5].set_ylabel('Frequency', fontsize=12)
        axes[5].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Image cleaning visualization saved to: {save_path}")
    return save_path

def create_sample_images_grid(image_paths, title, save_path, grid_size=(3, 3)):
    """Create a grid of sample images"""
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes):
        if i < len(image_paths) and os.path.exists(image_paths[i]):
            try:
                img = Image.open(image_paths[i])
                ax.imshow(img)
                ax.set_title(os.path.basename(image_paths[i]), fontsize=10)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{os.path.basename(image_paths[i])}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def visualize_quality_metrics_comparison(original_df, cleaned_df, save_path):
    """Compare quality metrics before and after cleaning"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    fig.suptitle('DataDome Image Quality Metrics: Before vs After Cleaning', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Blur Score Comparison
    axes[0].hist(original_df['blur_score'], bins=30, alpha=0.6, label='Original', 
                color='#FF6B6B', edgecolor='black', linewidth=0.5)
    axes[0].hist(cleaned_df['blur_score'], bins=30, alpha=0.6, label='Cleaned', 
                color='#4ECDC4', edgecolor='black', linewidth=0.5)
    axes[0].set_title('Blur Score Distribution', fontsize=12, pad=15)
    axes[0].set_xlabel('Blur Score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Brightness Comparison
    axes[1].hist(original_df['brightness'], bins=30, alpha=0.6, label='Original', 
                color='#FF6B6B', edgecolor='black', linewidth=0.5)
    axes[1].hist(cleaned_df['brightness'], bins=30, alpha=0.6, label='Cleaned', 
                color='#4ECDC4', edgecolor='black', linewidth=0.5)
    axes[1].set_title('Brightness Distribution', fontsize=12, pad=15)
    axes[1].set_xlabel('Brightness')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # 3. Resolution Comparison
    axes[2].hist(original_df['resolution'], bins=30, alpha=0.6, label='Original', 
                color='#FF6B6B', edgecolor='black', linewidth=0.5)
    axes[2].hist(cleaned_df['resolution'], bins=30, alpha=0.6, label='Cleaned', 
                color='#4ECDC4', edgecolor='black', linewidth=0.5)
    axes[2].set_title('Resolution Distribution', fontsize=12, pad=15)
    axes[2].set_xlabel('Resolution (pixels)')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    # 4. Contrast Comparison
    axes[3].hist(original_df['contrast'], bins=30, alpha=0.6, label='Original', 
                color='#FF6B6B', edgecolor='black', linewidth=0.5)
    axes[3].hist(cleaned_df['contrast'], bins=30, alpha=0.6, label='Cleaned', 
                color='#4ECDC4', edgecolor='black', linewidth=0.5)
    axes[3].set_title('Contrast Distribution', fontsize=12, pad=15)
    axes[3].set_xlabel('Contrast')
    axes[3].set_ylabel('Frequency')
    axes[3].legend()
    axes[3].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Quality metrics comparison saved to: {save_path}")
    return save_path