import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ImageDatasetModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.results = {}
        
    def extract_features_from_images(self, image_dir, max_images=None, csv_labels_df=None, target_column=None):
        """
        Extract features from images for ML training
        Returns: features array, labels array, filenames
        """
        features = []
        labels = []
        filenames = []
        
        print(f"Extracting features from images in {image_dir}...")
        
        # Get all image files
        image_files = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    image_files.append(os.path.join(root, file))
        
        # Limit number of images if specified
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
        
        print(f"Processing {len(image_files)} images...")
        
        # Determine label source
        use_csv_labels = csv_labels_df is not None and target_column is not None
        if use_csv_labels:
            print(f"📊 Using CSV labels for supervised learning")
            # Assume first column contains image names
            name_column = csv_labels_df.columns[0]
            print(f"🎯 Target column: {target_column}")
            
            # Validate target column exists
            if target_column not in csv_labels_df.columns:
                print(f"⚠️ Target column '{target_column}' not found in CSV, falling back to synthetic labels")
                use_csv_labels = False
        
        for i, image_path in enumerate(image_files):
            try:
                # Extract features from image
                img_features = self._extract_single_image_features(image_path)
                if img_features is not None:
                    features.append(img_features)
                    
                    # Get label from CSV or generate synthetic
                    filename = os.path.basename(image_path)
                    filename_no_ext = os.path.splitext(filename)[0].lower()
                    
                    if use_csv_labels:
                        # Try to find matching row in CSV
                        try:
                            matching_rows = csv_labels_df[csv_labels_df[name_column].str.lower() == filename_no_ext]
                            if not matching_rows.empty:
                                label = matching_rows.iloc[0][target_column]
                                # Handle NaN values
                                if pd.isna(label) or label == '' or str(label).lower() == 'nan':
                                    label = 'Unknown'
                            else:
                                # Try without file extension matching (in case CSV has extensions)
                                matching_rows = csv_labels_df[csv_labels_df[name_column].str.lower().str.contains(filename_no_ext)]
                                if not matching_rows.empty:
                                    label = matching_rows.iloc[0][target_column]
                                    if pd.isna(label) or label == '' or str(label).lower() == 'nan':
                                        label = 'Unknown'
                                else:
                                    label = 'Unknown'  # No match found in CSV
                        except Exception as e:
                            print(f"Error matching {filename} to CSV: {e}")
                            label = 'Unknown'
                    else:
                        # Generate synthetic labels based on image characteristics
                        label = self._generate_synthetic_label(img_features)
                    
                    labels.append(str(label))
                    filenames.append(filename)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if not features:
            return None, None, None
        
        # Print label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"📊 Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")
        
        # Check if we have enough samples for ML
        if len(features) < 10:
            print("⚠️ Very few samples for ML training. Consider uploading more images.")
        
        # Check for highly imbalanced classes
        max_count = max(counts)
        min_count = min(counts)
        if max_count / min_count > 10:
            print(f"⚠️ Highly imbalanced dataset. Ratio: {max_count}:{min_count}")
        
        # Check CSV matching success rate if using CSV labels
        if use_csv_labels:
            unknown_count = sum(1 for label in labels if str(label).lower() in ['unknown', 'nan'])
            match_rate = (len(labels) - unknown_count) / len(labels) * 100
            print(f"📊 CSV matching success rate: {match_rate:.1f}%")
            if match_rate < 50:
                print("⚠️ Low CSV matching rate. Check that image filenames match CSV entries.")
            
        return np.array(features), np.array(labels), filenames
    
    def _extract_single_image_features(self, image_path):
        """Extract comprehensive features from a single image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size for consistency
            img_resized = cv2.resize(img_rgb, (64, 64))
            
            features = []
            
            # 1. Color features
            # Mean color values
            mean_colors = np.mean(img_resized, axis=(0, 1))
            features.extend(mean_colors)
            
            # Color standard deviation
            std_colors = np.std(img_resized, axis=(0, 1))
            features.extend(std_colors)
            
            # 2. Texture features using Gray-Level Co-occurrence Matrix (simplified)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            
            # Contrast
            contrast = gray.std()
            features.append(contrast)
            
            # Homogeneity (inverse of contrast)
            homogeneity = 1.0 / (1.0 + contrast)
            features.append(homogeneity)
            
            # 3. Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 4. Histogram features
            hist_r = cv2.calcHist([img_resized], [0], None, [8], [0, 256])
            hist_g = cv2.calcHist([img_resized], [1], None, [8], [0, 256])
            hist_b = cv2.calcHist([img_resized], [2], None, [8], [0, 256])
            
            # Normalize histograms
            hist_r = hist_r.flatten() / np.sum(hist_r)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_b = hist_b.flatten() / np.sum(hist_b)
            
            features.extend(hist_r)
            features.extend(hist_g)
            features.extend(hist_b)
            
            # 5. Shape features
            # Aspect ratio
            aspect_ratio = img.shape[1] / img.shape[0]
            features.append(aspect_ratio)
            
            # Area (normalized)
            area = (img.shape[0] * img.shape[1]) / (512 * 512)  # Normalize to typical size
            features.append(area)
            
            # 6. Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(blur_score)
            
            # 7. Brightness
            brightness = np.mean(gray)
            features.append(brightness)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Feature extraction error for {image_path}: {e}")
            return None
    
    def _generate_synthetic_label(self, features):
        """
        Generate synthetic labels based on image features
        Creates a more challenging classification task by combining multiple features
        """
        # Extract key features
        brightness = features[-1]  # Last feature is brightness
        contrast = features[6] if len(features) > 6 else 50  # Contrast feature
        blur_score = features[-2] if len(features) > 1 else 100  # Blur score
        
        # Create more nuanced categories based on multiple features
        # This makes the task harder and more realistic
        
        # Calculate a quality score
        quality_score = 0
        
        # Brightness contribution (prefer moderate brightness)
        if 100 < brightness < 160:
            quality_score += 2
        elif 80 < brightness < 180:
            quality_score += 1
        
        # Contrast contribution
        if contrast > 60:
            quality_score += 2
        elif contrast > 40:
            quality_score += 1
        
        # Blur contribution (higher blur score = sharper)
        if blur_score > 200:
            quality_score += 2
        elif blur_score > 100:
            quality_score += 1
        
        # Classify based on combined quality score
        if quality_score >= 5:
            return 'High_Quality'
        elif quality_score >= 3:
            return 'Medium_Quality'
        elif quality_score >= 1:
            return 'Low_Quality'
        else:
            return 'Poor_Quality'
    
    def train_models(self, features, labels):
        """Train multiple ML models on the features"""
        print(f"Training models on {len(features)} samples with {len(features[0])} features...")
        
        # Check class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples = min(counts)
        
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")
        print(f"Minimum samples per class: {min_samples}")
        
        # Handle classes with too few samples
        if min_samples < 2:
            print("⚠️ Some classes have only 1 sample. Filtering out singleton classes...")
            # Remove classes with only 1 sample
            valid_indices = []
            for i, label in enumerate(labels):
                label_count = counts[unique_labels == label][0]
                if label_count >= 2:
                    valid_indices.append(i)
            
            if len(valid_indices) < 10:  # Need at least 10 samples total
                print("❌ Too few samples after filtering. Using random split instead of stratified.")
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )
            else:
                # Filter data to only include classes with >= 2 samples
                features_filtered = features[valid_indices]
                labels_filtered = labels[valid_indices]
                print(f"Filtered dataset: {len(features_filtered)} samples")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    features_filtered, labels_filtered, test_size=0.2, random_state=42, stratify=labels_filtered
                )
        else:
            # Normal stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'classification_report': report,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'model': model
                }
                
                print(f"{model_name} - Accuracy: {accuracy:.3f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def compare_datasets(self, original_dir, cleaned_dir, max_images_per_dataset=500, csv_labels_path=None, target_column=None):
        """
        Compare model performance on original vs cleaned datasets
        """
        print("=== Comparing Original vs Cleaned Dataset Performance ===")
        
        # Load CSV labels if provided
        csv_labels_df = None
        if csv_labels_path and os.path.exists(csv_labels_path):
            try:
                csv_labels_df = pd.read_csv(csv_labels_path)
                print(f"📊 Loaded CSV labels: {len(csv_labels_df)} rows, columns: {csv_labels_df.columns.tolist()}")
                if target_column:
                    print(f"🎯 Using target column: {target_column}")
            except Exception as e:
                print(f"⚠️ Error loading CSV labels: {e}")
                csv_labels_df = None
        
        # Extract features from original dataset
        print("\n1. Processing Original Dataset...")
        orig_features, orig_labels, orig_files = self.extract_features_from_images(
            original_dir, max_images_per_dataset, csv_labels_df, target_column
        )
        
        if orig_features is None:
            print("❌ No features extracted from original dataset")
            return None
        
        # Extract features from cleaned dataset
        print("\n2. Processing Cleaned Dataset...")
        clean_features, clean_labels, clean_files = self.extract_features_from_images(
            cleaned_dir, max_images_per_dataset, csv_labels_df, target_column
        )
        
        if clean_features is None:
            print("❌ No features extracted from cleaned dataset")
            return None
        
        # Check if we have enough samples for meaningful ML training
        if len(orig_features) < 20 or len(clean_features) < 20:
            print("⚠️ Too few samples for reliable ML training (minimum 20 recommended)")
            print("Consider uploading more images or reducing max_images_for_ml parameter")
            return None
        
        # Train models on original dataset
        print("\n3. Training Models on Original Dataset...")
        original_results = self.train_models(orig_features, orig_labels)
        
        # Train models on cleaned dataset
        print("\n4. Training Models on Cleaned Dataset...")
        cleaned_results = self.train_models(clean_features, clean_labels)
        
        # Compare results
        print("\n5. Comparing Results...")
        comparison = self._compare_results(original_results, cleaned_results)
        
        return {
            'original_results': original_results,
            'cleaned_results': cleaned_results,
            'comparison': comparison,
            'original_samples': len(orig_features),
            'cleaned_samples': len(clean_features)
        }
    
    def _compare_results(self, original_results, cleaned_results):
        """Compare results between original and cleaned datasets"""
        comparison = {}
        
        for model_name in self.models.keys():
            if (model_name in original_results and original_results[model_name] and
                model_name in cleaned_results and cleaned_results[model_name]):
                
                orig_acc = original_results[model_name]['accuracy']
                clean_acc = cleaned_results[model_name]['accuracy']
                improvement = clean_acc - orig_acc
                improvement_pct = (improvement / orig_acc) * 100 if orig_acc > 0 else 0
                
                comparison[model_name] = {
                    'original_accuracy': orig_acc,
                    'cleaned_accuracy': clean_acc,
                    'improvement': improvement,
                    'improvement_percentage': improvement_pct
                }
        
        return comparison
    
    def visualize_comparison(self, comparison_results, save_path):
        """Create visualization comparing model performance"""
        if not comparison_results or 'comparison' not in comparison_results:
            print("No comparison results to visualize")
            return None
        
        comparison = comparison_results['comparison']
        
        # Create larger figure with better spacing
        fig, axes = plt.subplots(2, 2, figsize=(18, 15))
        axes = axes.flatten()
        
        # Main title with better formatting
        fig.suptitle('🤖 DataDome Image Dataset: ML Model Performance Comparison', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Accuracy Comparison
        models = list(comparison.keys())
        original_accs = [comparison[model]['original_accuracy'] for model in models]
        cleaned_accs = [comparison[model]['cleaned_accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, original_accs, width, label='Original Dataset', 
                           color='#FF6B6B', alpha=0.8, edgecolor='darkred', linewidth=1.5)
        bars2 = axes[0].bar(x + width/2, cleaned_accs, width, label='Cleaned Dataset', 
                           color='#4ECDC4', alpha=0.8, edgecolor='darkcyan', linewidth=1.5)
        
        axes[0].set_title('📊 Model Accuracy Comparison', fontsize=14, pad=20, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        axes[0].legend(fontsize=11, framealpha=0.9)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].set_ylim([0, 1.1])  # Set consistent y-axis
        
        # Add value labels on bars with better positioning
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Improvement Percentage with better color coding
        improvements = [comparison[model]['improvement_percentage'] for model in models]
        colors = ['#27ae60' if imp > 0.1 else '#e74c3c' if imp < -0.1 else '#95a5a6' for imp in improvements]
        
        bars = axes[1].bar(models, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_title('📈 Performance Improvement (%)', fontsize=14, pad=20, fontweight='bold')
        axes[1].set_ylabel('Improvement Percentage', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        
        # Add value labels with better positioning
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            label_y = height + (1 if height > 0 else -2)
            axes[1].text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{imp:.2f}%', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
        
        # 3. Dataset Size Comparison with better visualization
        dataset_sizes = [
            comparison_results['original_samples'],
            comparison_results['cleaned_samples']
        ]
        dataset_labels = ['Original\nDataset', 'Cleaned\nDataset']
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = axes[2].bar(dataset_labels, dataset_sizes, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=2, width=0.6)
        axes[2].set_title('📦 Dataset Size Comparison', fontsize=14, pad=20, fontweight='bold')
        axes[2].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels with better formatting
        for bar, size in zip(bars, dataset_sizes):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + max(dataset_sizes)*0.02,
                        f'{size:,}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # Add reduction percentage
        if dataset_sizes[0] > 0:
            reduction_pct = ((dataset_sizes[0] - dataset_sizes[1]) / dataset_sizes[0]) * 100
            axes[2].text(0.5, 0.95, f'Reduction: {reduction_pct:.1f}%', 
                        transform=axes[2].transAxes, ha='center', va='top',
                        fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 4. Enhanced Performance Summary
        avg_improvement = np.mean(improvements)
        best_model = models[np.argmax(improvements)]
        best_improvement = max(improvements)
        worst_model = models[np.argmin(improvements)]
        worst_improvement = min(improvements)
        
        # Clear axis and create text summary
        axes[3].axis('off')
        
        # Title
        axes[3].text(0.5, 0.95, '📋 Performance Summary', fontsize=16, fontweight='bold', 
                    transform=axes[3].transAxes, ha='center', va='top')
        
        # Summary statistics with better formatting
        summary_lines = [
            f"{'='*50}",
            f"",
            f"📊 Overall Performance:",
            f"   Average Improvement: {avg_improvement:.2f}%",
            f"   Models Evaluated: {len(models)}",
            f"",
            f"🏆 Best Performing Model:",
            f"   {best_model}",
            f"   Improvement: {best_improvement:.2f}%",
            f"",
            f"📉 Worst Performing Model:",
            f"   {worst_model}",
            f"   Change: {worst_improvement:.2f}%",
            f"",
            f"📦 Dataset Statistics:",
            f"   Original: {comparison_results['original_samples']:,} images",
            f"   Cleaned: {comparison_results['cleaned_samples']:,} images",
            f"   Removed: {comparison_results['original_samples'] - comparison_results['cleaned_samples']:,} images",
            f"",
            f"🔧 Processing Mode:",
            f"   {'Enhanced Quality' if 'enhanced' in str(save_path).lower() else 'Cleaning Only'}",
            f"",
            f"{'='*50}"
        ]
        
        summary_text = '\n'.join(summary_lines)
        
        axes[3].text(0.05, 0.85, summary_text, fontsize=10, transform=axes[3].transAxes,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle="round,pad=0.8", facecolor='#f0f0f0', 
                             edgecolor='#333333', linewidth=2, alpha=0.9))
        
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.35, wspace=0.3)
        
        # Save with high quality
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"✅ Model comparison visualization saved to: {save_path}")
        return save_path