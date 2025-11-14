import os
import pandas as pd
from pre_processing.modules.image_processing import ImageDatasetCleaner
from pre_processing.modules.image_visualization import (
    visualize_image_cleaning_results, 
    create_sample_images_grid,
    visualize_quality_metrics_comparison
)
from pre_processing.modules.image_enhancement import ImageEnhancer, calculate_quality_improvement
from pre_processing.modules.image_models import ImageDatasetModelTrainer

def process_image_dataset(dataset_path, 
                         min_resolution=32*32,
                         max_resolution=4096*4096,
                         min_blur_threshold=100,
                         min_brightness=10,
                         max_brightness=245,
                         min_contrast=10,
                         remove_duplicates=True,
                         remove_corrupted=True,
                         enhance_images=False,
                         enhancement_options=None,
                         train_models=False,
                         max_images_for_ml=500,
                         csv_labels_path=None,
                         target_column=None):
    """
    Main function to process and clean an image dataset
    
    Args:
        dataset_path: Path to the image dataset directory
        min_resolution: Minimum acceptable resolution (width * height)
        max_resolution: Maximum acceptable resolution (width * height)
        min_blur_threshold: Minimum blur score (higher = sharper)
        min_brightness: Minimum acceptable brightness
        max_brightness: Maximum acceptable brightness
        min_contrast: Minimum acceptable contrast
        remove_duplicates: Whether to remove duplicate images
        remove_corrupted: Whether to remove corrupted images
    
    Returns:
        tuple: (original_df, cleaned_df, cleaning_stats, quality_report)
    """
    
    print(f"Starting DataDome image dataset processing...")
    print(f"Dataset path: {dataset_path}")
    
    # Initialize the image cleaner
    cleaner = ImageDatasetCleaner(dataset_path)
    
    # Step 1: Analyze the dataset
    print("\n=== Step 1: Analyzing Dataset ===")
    original_df = cleaner.analyze_dataset()
    
    if original_df.empty:
        print("No valid images found in the dataset!")
        return None, None, None, None
    
    print(f"Found {len(original_df)} images in the dataset")
    
    # Step 2: Generate quality report
    print("\n=== Step 2: Generating Quality Report ===")
    quality_report = cleaner.generate_quality_report()
    
    print("Quality Report:")
    print(f"  Total images: {quality_report['total_images']}")
    print(f"  Corrupted images: {quality_report['corrupted_images']}")
    print(f"  Average resolution: {quality_report['avg_resolution']:.0f} pixels")
    print(f"  Average file size: {quality_report['avg_file_size']/1024/1024:.2f} MB")
    print(f"  Average blur score: {quality_report['avg_blur_score']:.2f}")
    print(f"  Average brightness: {quality_report['avg_brightness']:.2f}")
    print(f"  Average contrast: {quality_report['avg_contrast']:.2f}")
    print(f"  Duplicate images: {quality_report['duplicate_count']}")
    
    # Step 3: Clean the dataset
    print("\n=== Step 3: Cleaning Dataset ===")
    cleaned_df, cleaning_stats = cleaner.clean_dataset(
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        min_blur_threshold=min_blur_threshold,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        min_contrast=min_contrast,
        remove_duplicates=remove_duplicates,
        remove_corrupted=remove_corrupted
    )
    
    # Step 4: Generate visualizations
    print("\n=== Step 4: Generating Visualizations ===")
    
    # Create output directory for visualizations
    viz_dir = "app/static/image_results"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Main cleaning results visualization
    main_viz_path = os.path.join(viz_dir, "image_cleaning_results.png")
    visualize_image_cleaning_results(
        original_df, cleaned_df, cleaning_stats, main_viz_path
    )
    
    # Quality metrics comparison
    if len(cleaned_df) > 0:
        quality_viz_path = os.path.join(viz_dir, "quality_metrics_comparison.png")
        visualize_quality_metrics_comparison(
            original_df, cleaned_df, quality_viz_path
        )
    
    # Sample images grid (before cleaning)
    if len(original_df) > 0:
        sample_original_paths = original_df['file_path'].head(9).tolist()
        original_grid_path = os.path.join(viz_dir, "sample_original_images.png")
        create_sample_images_grid(
            sample_original_paths, 
            "Sample Original Images", 
            original_grid_path
        )
    
    # Sample images grid (after cleaning)
    if len(cleaned_df) > 0:
        sample_cleaned_paths = cleaned_df['file_path'].head(9).tolist()
        cleaned_grid_path = os.path.join(viz_dir, "sample_cleaned_images.png")
        create_sample_images_grid(
            sample_cleaned_paths, 
            "Sample Cleaned Images", 
            cleaned_grid_path
        )
    
    # Step 5: Image Enhancement (Optional)
    enhancement_stats = None
    quality_improvements = None
    
    if enhance_images and len(cleaned_df) > 0:
        print("\n=== Step 5: Enhancing Images ===")
        
        # Default enhancement options
        if enhancement_options is None:
            enhancement_options = {
                'auto_brightness': True,
                'enhance_contrast': True,
                'sharpen': True,
                'denoise': True,
                'standardize_size': False,
                'standardize_format': True,
                'target_format': 'JPEG'
            }
        
        # Create enhanced images directory
        enhanced_dir = "output/enhanced_images"
        os.makedirs(enhanced_dir, exist_ok=True)
        
        # Enhance the cleaned images
        enhancer = ImageEnhancer()
        enhanced_count, failed_count, enhancement_stats = enhancer.enhance_dataset(
            cleaner.cleaned_path, enhanced_dir, enhancement_options
        )
        
        print(f"Enhanced {enhanced_count} images, {failed_count} failed")
        print("Enhancement breakdown:")
        for key, value in enhancement_stats.items():
            if value > 0:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Calculate quality improvements
        print("\n=== Step 6: Calculating Quality Improvements ===")
        quality_improvements = calculate_quality_improvement(
            cleaner.cleaned_path, enhanced_dir
        )
        
        if quality_improvements:
            print("Quality improvements:")
            for key, value in quality_improvements.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
    
    # Step 6/7/8: ML Model Training and Comparison (Optional)
    model_comparison_results = None
    
    if train_models and len(cleaned_df) > 0:
        step_num = 6 if not enhance_images else 7
        print(f"\n=== Step {step_num}: ML Model Training & Comparison ===")
        
        # Initialize model trainer
        trainer = ImageDatasetModelTrainer()
        
        # Compare original vs cleaned dataset performance
        original_dataset_path = dataset_path
        cleaned_dataset_path = cleaner.cleaned_path
        
        # Use enhanced images if available, otherwise cleaned images
        comparison_dataset_path = "output/enhanced_images" if enhance_images and os.path.exists("output/enhanced_images") else cleaned_dataset_path
        
        model_comparison_results = trainer.compare_datasets(
            original_dataset_path, 
            comparison_dataset_path,
            max_images_per_dataset=max_images_for_ml,
            csv_labels_path=csv_labels_path,
            target_column=target_column
        )
        
        if model_comparison_results:
            # Create model comparison visualization
            model_viz_path = os.path.join(viz_dir, "model_performance_comparison.png")
            trainer.visualize_comparison(model_comparison_results, model_viz_path)
            
            # Save model results
            if model_comparison_results['comparison']:
                comparison_df = pd.DataFrame(model_comparison_results['comparison']).T
                comparison_df.to_csv("output/model_performance_comparison.csv")
                print("Model performance comparison saved to: output/model_performance_comparison.csv")
        
        step_num += 1
    else:
        step_num = 6 if not enhance_images else 7
    
    # Final step: Save results to CSV
    print(f"\n=== Step {step_num}: Saving Results ===")
    
    # Save original analysis
    original_df.to_csv("output/original_image_analysis.csv", index=False)
    print("Original image analysis saved to: output/original_image_analysis.csv")
    
    # Save cleaned dataset info
    if len(cleaned_df) > 0:
        cleaned_df.to_csv("output/cleaned_image_analysis.csv", index=False)
        print("Cleaned image analysis saved to: output/cleaned_image_analysis.csv")
    
    # Save cleaning statistics
    cleaning_stats_df = pd.DataFrame([cleaning_stats])
    cleaning_stats_df.to_csv("output/image_cleaning_stats.csv", index=False)
    print("Cleaning statistics saved to: output/image_cleaning_stats.csv")
    
    # Save enhancement statistics if available
    if enhancement_stats:
        enhancement_df = pd.DataFrame([enhancement_stats])
        enhancement_df.to_csv("output/image_enhancement_stats.csv", index=False)
        print("Enhancement statistics saved to: output/image_enhancement_stats.csv")
    
    # Save quality improvements if available
    if quality_improvements:
        improvements_df = pd.DataFrame([quality_improvements])
        improvements_df.to_csv("output/image_quality_improvements.csv", index=False)
        print("Quality improvements saved to: output/image_quality_improvements.csv")
    
    print(f"\n=== DataDome Image Processing Complete ===")
    print(f"Cleaned images saved to: {cleaner.cleaned_path}")
    if enhance_images:
        print(f"Enhanced images saved to: output/enhanced_images/")
    if train_models and model_comparison_results:
        print(f"ML model comparison results saved to: output/model_performance_comparison.csv")
    print(f"Visualizations saved to: {viz_dir}")
    
    return original_df, cleaned_df, cleaning_stats, quality_report, enhancement_stats, quality_improvements, model_comparison_results

def get_image_processing_summary(cleaning_stats, quality_report, enhancement_stats=None, quality_improvements=None, model_comparison_results=None):
    """Generate a summary of the image processing results"""
    
    if not cleaning_stats or not quality_report:
        return "No processing results available"
    
    initial_count = cleaning_stats['initial_count']
    final_count = cleaning_stats['final_count']
    removed_count = initial_count - final_count
    removal_percentage = (removed_count / initial_count) * 100 if initial_count > 0 else 0
    
    summary = f"""
DataDome Image Dataset Processing Summary:

📊 Dataset Overview:
   • Initial images: {initial_count:,}
   • Final images: {final_count:,}
   • Images removed: {removed_count:,} ({removal_percentage:.1f}%)

🧹 Cleaning Breakdown:
   • Corrupted images: {cleaning_stats['corrupted_removed']}
   • Low resolution: {cleaning_stats['low_resolution_removed']}
   • High resolution: {cleaning_stats['high_resolution_removed']}
   • Blurry images: {cleaning_stats['blurry_removed']}
   • Brightness issues: {cleaning_stats['brightness_issues_removed']}
   • Low contrast: {cleaning_stats['low_contrast_removed']}
   • Duplicates: {cleaning_stats['duplicates_removed']}

📈 Quality Metrics (Original Dataset):
   • Average resolution: {quality_report['avg_resolution']:.0f} pixels
   • Average file size: {quality_report['avg_file_size']/1024/1024:.2f} MB
   • Average blur score: {quality_report['avg_blur_score']:.2f}
   • Average brightness: {quality_report['avg_brightness']:.2f}
   • Average contrast: {quality_report['avg_contrast']:.2f}"""

    # Add enhancement information if available
    if enhancement_stats:
        summary += f"""

🚀 Image Enhancement Applied:
   • Brightness adjusted: {enhancement_stats['brightness_adjusted']}
   • Contrast enhanced: {enhancement_stats['contrast_enhanced']}
   • Sharpness improved: {enhancement_stats['sharpness_improved']}
   • Noise reduced: {enhancement_stats['noise_reduced']}
   • Format standardized: {enhancement_stats['format_standardized']}"""

    if quality_improvements:
        summary += f"""

📈 Quality Improvements:
   • Average brightness change: {quality_improvements['avg_brightness_change']:.2f}
   • Average contrast change: {quality_improvements['avg_contrast_change']:.2f}
   • Average sharpness change: {quality_improvements['avg_blur_change']:.2f}
   • Images with better contrast: {quality_improvements['contrast_improvement']:.1f}%
   • Images with better sharpness: {quality_improvements['sharpness_improvement']:.1f}%"""

    summary += f"""

✅ Results:
   • Cleaned images saved to: output/cleaned_images/"""
    
    if enhancement_stats:
        summary += f"""
   • Enhanced images saved to: output/enhanced_images/"""
    
    # Add ML model comparison results if available
    if model_comparison_results and model_comparison_results.get('comparison'):
        comparison = model_comparison_results['comparison']
        avg_improvement = sum(comp['improvement_percentage'] for comp in comparison.values()) / len(comparison)
        best_model = max(comparison.keys(), key=lambda k: comparison[k]['improvement_percentage'])
        best_improvement = comparison[best_model]['improvement_percentage']
        
        summary += f"""

🤖 ML Model Performance Comparison:
   • Models trained: {len(comparison)}
   • Average accuracy improvement: {avg_improvement:.1f}%
   • Best performing model: {best_model}
   • Best improvement: {best_improvement:.1f}%
   • Training samples: {model_comparison_results['original_samples']} → {model_comparison_results['cleaned_samples']}"""
    
    summary += f"""
   • Analysis reports saved to: output/"""
    
    if model_comparison_results:
        summary += f"""
   • ML comparison results saved to: output/model_performance_comparison.csv"""
    
    summary += f"""
   • Visualizations saved to: app/static/image_results/
"""
    
    return summary