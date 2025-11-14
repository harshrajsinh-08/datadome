import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shutil
from pathlib import Path

class ImageDatasetCleaner:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_info = []
        self.cleaned_path = "output/cleaned_images"
        self.stats = {}
        
    def analyze_dataset(self):
        """Analyze the image dataset for quality issues"""
        print("Analyzing image dataset...")
        
        # Create output directory
        os.makedirs(self.cleaned_path, exist_ok=True)
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Walk through dataset directory
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext in supported_formats:
                    try:
                        # Analyze each image
                        image_stats = self._analyze_single_image(file_path)
                        if image_stats:
                            self.image_info.append(image_stats)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.image_info)
        return self.df
    
    def _analyze_single_image(self, image_path):
        """Analyze a single image for quality metrics"""
        try:
            # Load image with PIL for basic stats
            pil_img = Image.open(image_path)
            
            # Load with OpenCV for advanced analysis
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                return None
                
            # Basic image properties
            width, height = pil_img.size
            channels = len(pil_img.getbands())
            file_size = os.path.getsize(image_path)
            
            # Image quality metrics
            blur_score = self._calculate_blur(cv_img)
            brightness = self._calculate_brightness(pil_img)
            contrast = self._calculate_contrast(pil_img)
            
            # Check for corruption
            is_corrupted = self._check_corruption(pil_img, cv_img, image_path)
            
            # Check for duplicates (using image hash)
            image_hash = self._calculate_hash(cv_img)
            
            return {
                'file_path': image_path,
                'filename': os.path.basename(image_path),
                'width': width,
                'height': height,
                'channels': channels,
                'file_size': file_size,
                'aspect_ratio': width / height,
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast,
                'is_corrupted': is_corrupted,
                'image_hash': image_hash,
                'resolution': width * height
            }
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def _calculate_blur(self, image):
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calculate_brightness(self, image):
        """Calculate average brightness"""
        stat = ImageStat.Stat(image)
        return sum(stat.mean) / len(stat.mean)
    
    def _calculate_contrast(self, image):
        """Calculate contrast using standard deviation"""
        stat = ImageStat.Stat(image)
        return sum(stat.stddev) / len(stat.stddev)
    
    def _check_corruption(self, pil_img, cv_img, image_path):
        """Comprehensive corruption detection"""
        corruption_reasons = []
        
        try:
            # Test 1: Check if OpenCV could load the image
            if cv_img is None:
                corruption_reasons.append("OpenCV failed to load")
            
            # Test 2: Check if PIL image has valid dimensions
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                corruption_reasons.append("Invalid dimensions")
            
            # Test 3: Check if image has valid data
            if cv_img is not None and cv_img.size == 0:
                corruption_reasons.append("Empty image data")
            
            # Test 4: Try to verify PIL image integrity
            try:
                # Create a copy to avoid consuming the original
                test_img = Image.open(image_path)
                test_img.verify()
            except Exception as e:
                corruption_reasons.append(f"PIL verification failed: {str(e)}")
            
            # Test 5: Check for extremely small file size (likely truncated)
            file_size = os.path.getsize(image_path)
            if file_size < 100:  # Less than 100 bytes is suspicious
                corruption_reasons.append("File too small (likely truncated)")
            
            # Test 6: Check for valid image statistics
            if cv_img is not None:
                try:
                    # Calculate basic statistics - corrupted images often have unusual stats
                    mean_val = cv_img.mean()
                    std_val = cv_img.std()
                    
                    # Check for completely black or white images (possible corruption)
                    if mean_val < 1 or mean_val > 254:
                        corruption_reasons.append("Suspicious pixel values (all black/white)")
                    
                    # Check for zero standard deviation (flat image, possibly corrupted)
                    if std_val < 0.1:
                        corruption_reasons.append("Zero variance (flat image)")
                        
                except Exception as e:
                    corruption_reasons.append(f"Statistics calculation failed: {str(e)}")
            
            # Test 7: Check aspect ratio sanity
            if pil_img.size[0] > 0 and pil_img.size[1] > 0:
                aspect_ratio = pil_img.size[0] / pil_img.size[1]
                if aspect_ratio > 100 or aspect_ratio < 0.01:
                    corruption_reasons.append("Extreme aspect ratio")
            
            # Test 8: Try to read pixel data
            try:
                if cv_img is not None and cv_img.shape[0] > 0 and cv_img.shape[1] > 0:
                    # Try to access a pixel
                    _ = cv_img[0, 0]
            except Exception as e:
                corruption_reasons.append(f"Pixel access failed: {str(e)}")
            
            # Log corruption reasons for debugging
            if corruption_reasons:
                print(f"Corruption detected in {os.path.basename(image_path)}: {', '.join(corruption_reasons)}")
                return True
            
            return False
            
        except Exception as e:
            corruption_reasons.append(f"General error: {str(e)}")
            print(f"Corruption check error for {os.path.basename(image_path)}: {str(e)}")
            return True
    
    def _calculate_hash(self, image):
        """Calculate perceptual hash for duplicate detection"""
        # Resize to 8x8 and convert to grayscale
        resized = cv2.resize(image, (8, 8))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate average
        avg = gray.mean()
        
        # Create hash
        hash_str = ""
        for row in gray:
            for pixel in row:
                hash_str += "1" if pixel > avg else "0"
        
        return hash_str
    
    def clean_dataset(self, 
                     min_resolution=32*32,
                     max_resolution=4096*4096,
                     min_blur_threshold=100,
                     min_brightness=10,
                     max_brightness=245,
                     min_contrast=10,
                     remove_duplicates=True,
                     remove_corrupted=True):
        """Clean the dataset based on quality thresholds"""
        
        if self.df is None or self.df.empty:
            print("No image data to clean. Run analyze_dataset() first.")
            return
        
        print("Cleaning image dataset...")
        
        # Initial count
        initial_count = len(self.df)
        
        # Show corruption detection summary
        corrupted_count = (self.df['is_corrupted'] == True).sum()
        print(f"Corruption detection summary:")
        print(f"  Total images analyzed: {initial_count}")
        print(f"  Corrupted images found: {corrupted_count}")
        print(f"  Corruption rate: {(corrupted_count/initial_count)*100:.1f}%")
        
        # Track cleaning steps
        cleaning_stats = {
            'initial_count': initial_count,
            'corrupted_removed': 0,
            'low_resolution_removed': 0,
            'high_resolution_removed': 0,
            'blurry_removed': 0,
            'brightness_issues_removed': 0,
            'low_contrast_removed': 0,
            'duplicates_removed': 0,
            'final_count': 0
        }
        
        # Create a copy for cleaning
        cleaned_df = self.df.copy()
        
        # Remove corrupted images
        if remove_corrupted:
            corrupted_mask = cleaned_df['is_corrupted'] == True
            corrupted_count = int(corrupted_mask.sum())
            cleaning_stats['corrupted_removed'] = corrupted_count
            print(f"Debug: Found {corrupted_count} corrupted images out of {len(cleaned_df)} total")
            if corrupted_count > 0:
                print("Debug: Sample of corrupted images:")
                corrupted_samples = cleaned_df[corrupted_mask]['filename'].head(5).tolist()
                print(f"  {corrupted_samples}")
            cleaned_df = cleaned_df[~corrupted_mask]
        
        # Remove images with resolution issues
        resolution_mask = (cleaned_df['resolution'] < min_resolution) | (cleaned_df['resolution'] > max_resolution)
        cleaning_stats['low_resolution_removed'] = int((cleaned_df['resolution'] < min_resolution).sum())
        cleaning_stats['high_resolution_removed'] = int((cleaned_df['resolution'] > max_resolution).sum())
        cleaned_df = cleaned_df[~resolution_mask]
        
        # Remove blurry images
        blur_mask = cleaned_df['blur_score'] < min_blur_threshold
        cleaning_stats['blurry_removed'] = int(blur_mask.sum())
        cleaned_df = cleaned_df[~blur_mask]
        
        # Remove images with brightness issues
        brightness_mask = (cleaned_df['brightness'] < min_brightness) | (cleaned_df['brightness'] > max_brightness)
        cleaning_stats['brightness_issues_removed'] = int(brightness_mask.sum())
        cleaned_df = cleaned_df[~brightness_mask]
        
        # Remove low contrast images
        contrast_mask = cleaned_df['contrast'] < min_contrast
        cleaning_stats['low_contrast_removed'] = int(contrast_mask.sum())
        cleaned_df = cleaned_df[~contrast_mask]
        
        # Remove duplicates
        if remove_duplicates:
            duplicate_mask = cleaned_df.duplicated(subset=['image_hash'], keep='first')
            cleaning_stats['duplicates_removed'] = int(duplicate_mask.sum())
            cleaned_df = cleaned_df[~duplicate_mask]
        
        cleaning_stats['final_count'] = int(len(cleaned_df))
        
        # Copy cleaned images to output directory
        self._copy_cleaned_images(cleaned_df)
        
        self.cleaned_df = cleaned_df
        self.cleaning_stats = cleaning_stats
        
        print(f"Dataset cleaning completed:")
        print(f"  Initial images: {initial_count}")
        print(f"  Final images: {cleaning_stats['final_count']}")
        print(f"  Removed: {initial_count - cleaning_stats['final_count']} images")
        
        return cleaned_df, cleaning_stats
    
    def _copy_cleaned_images(self, cleaned_df):
        """Copy cleaned images to output directory"""
        print("Copying cleaned images...")
        
        for _, row in cleaned_df.iterrows():
            src_path = row['file_path']
            dst_path = os.path.join(self.cleaned_path, row['filename'])
            
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
    
    def generate_quality_report(self):
        """Generate a comprehensive quality report"""
        if self.df is None or self.df.empty:
            return None
        
        report = {
            'total_images': int(len(self.df)),
            'corrupted_images': int(self.df['is_corrupted'].sum()),
            'avg_resolution': float(self.df['resolution'].mean()),
            'avg_file_size': float(self.df['file_size'].mean()),
            'avg_blur_score': float(self.df['blur_score'].mean()),
            'avg_brightness': float(self.df['brightness'].mean()),
            'avg_contrast': float(self.df['contrast'].mean()),
            'resolution_distribution': self.df['resolution'].describe().to_dict(),
            'aspect_ratio_distribution': self.df['aspect_ratio'].describe().to_dict(),
            'duplicate_count': int(len(self.df) - len(self.df['image_hash'].unique()))
        }
        
        return report