import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
import os

class ImageEnhancer:
    def __init__(self):
        self.enhancement_stats = {
            'brightness_adjusted': 0,
            'contrast_enhanced': 0,
            'sharpness_improved': 0,
            'noise_reduced': 0,
            'resized': 0,
            'format_standardized': 0
        }
    
    def enhance_image(self, image_path, output_path, enhancements):
        """
        Apply various enhancements to an image
        
        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image
            enhancements: Dict of enhancement settings
        """
        try:
            # Load image with PIL and OpenCV
            pil_img = Image.open(image_path)
            cv_img = cv2.imread(image_path)
            
            enhanced_img = pil_img.copy()
            
            # 1. Brightness adjustment
            if enhancements.get('auto_brightness', False):
                enhanced_img = self._adjust_brightness(enhanced_img)
                self.enhancement_stats['brightness_adjusted'] += 1
            
            # 2. Contrast enhancement
            if enhancements.get('enhance_contrast', False):
                enhanced_img = self._enhance_contrast(enhanced_img)
                self.enhancement_stats['contrast_enhanced'] += 1
            
            # 3. Sharpness improvement
            if enhancements.get('sharpen', False):
                enhanced_img = self._sharpen_image(enhanced_img)
                self.enhancement_stats['sharpness_improved'] += 1
            
            # 4. Noise reduction (using OpenCV)
            if enhancements.get('denoise', False):
                enhanced_img = self._reduce_noise(enhanced_img, cv_img)
                self.enhancement_stats['noise_reduced'] += 1
            
            # 5. Resize to standard dimensions
            if enhancements.get('standardize_size', False):
                target_size = enhancements.get('target_size', (224, 224))
                enhanced_img = self._resize_image(enhanced_img, target_size)
                self.enhancement_stats['resized'] += 1
            
            # 6. Format standardization
            if enhancements.get('standardize_format', False):
                target_format = enhancements.get('target_format', 'JPEG')
                enhanced_img = self._standardize_format(enhanced_img, target_format)
                self.enhancement_stats['format_standardized'] += 1
            
            # Save enhanced image
            if enhanced_img.mode == 'RGBA' and output_path.lower().endswith('.jpg'):
                # Convert RGBA to RGB for JPEG
                rgb_img = Image.new('RGB', enhanced_img.size, (255, 255, 255))
                rgb_img.paste(enhanced_img, mask=enhanced_img.split()[-1])
                enhanced_img = rgb_img
            
            enhanced_img.save(output_path, quality=95)
            return True
            
        except Exception as e:
            print(f"Error enhancing {image_path}: {e}")
            return False
    
    def _adjust_brightness(self, image):
        """Auto-adjust brightness based on image histogram"""
        # Calculate current brightness
        stat = ImageStat.Stat(image)
        current_brightness = sum(stat.mean) / len(stat.mean)
        
        # Target brightness (around 128 for 8-bit images)
        target_brightness = 128
        
        # Calculate adjustment factor
        if current_brightness < 50:  # Too dark
            factor = 1.3
        elif current_brightness > 200:  # Too bright
            factor = 0.8
        else:
            factor = target_brightness / current_brightness
            factor = max(0.7, min(1.5, factor))  # Limit adjustment range
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _enhance_contrast(self, image):
        """Enhance contrast using adaptive methods"""
        # Calculate current contrast (standard deviation)
        stat = ImageStat.Stat(image)
        current_contrast = sum(stat.stddev) / len(stat.stddev)
        
        # Enhance contrast if it's too low
        if current_contrast < 30:
            factor = 1.4
        elif current_contrast < 50:
            factor = 1.2
        else:
            factor = 1.1
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _sharpen_image(self, image):
        """Apply sharpening filter"""
        # Apply unsharp mask
        enhancer = ImageEnhance.Sharpness(image)
        sharpened = enhancer.enhance(1.3)
        
        # Additional edge enhancement
        edge_enhanced = sharpened.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Blend original and edge-enhanced
        return Image.blend(sharpened, edge_enhanced, 0.3)
    
    def _reduce_noise(self, pil_img, cv_img):
        """Reduce noise using OpenCV denoising"""
        if cv_img is None:
            return pil_img
        
        # Apply Non-local Means Denoising
        if len(cv_img.shape) == 3:  # Color image
            denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        else:  # Grayscale image
            denoised = cv2.fastNlMeansDenoising(cv_img, None, 10, 7, 21)
        
        # Convert back to PIL
        if len(denoised.shape) == 3:
            denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
            return Image.fromarray(denoised_rgb)
        else:
            return Image.fromarray(denoised)
    
    def _resize_image(self, image, target_size):
        """Resize image while maintaining aspect ratio"""
        # Calculate new size maintaining aspect ratio
        original_size = image.size
        ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
        
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        # Resize with high-quality resampling
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image
        final_img = Image.new('RGB', target_size, (255, 255, 255))
        paste_pos = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
        final_img.paste(resized, paste_pos)
        
        return final_img
    
    def _standardize_format(self, image, target_format):
        """Standardize image format"""
        if target_format.upper() == 'JPEG' and image.mode in ['RGBA', 'LA']:
            # Convert to RGB for JPEG
            rgb_img = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                rgb_img.paste(image, mask=image.split()[-1])
            else:
                rgb_img.paste(image)
            return rgb_img
        
        return image
    
    def enhance_dataset(self, input_dir, output_dir, enhancements):
        """
        Enhance all images in a dataset
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save enhanced images
            enhancements: Dict of enhancement settings
        """
        os.makedirs(output_dir, exist_ok=True)
        
        enhanced_count = 0
        failed_count = 0
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in supported_formats:
                    input_path = os.path.join(root, file)
                    
                    # Determine output filename
                    if enhancements.get('standardize_format', False):
                        target_format = enhancements.get('target_format', 'JPEG')
                        if target_format.upper() == 'JPEG':
                            output_file = os.path.splitext(file)[0] + '.jpg'
                        else:
                            output_file = file
                    else:
                        output_file = file
                    
                    output_path = os.path.join(output_dir, output_file)
                    
                    if self.enhance_image(input_path, output_path, enhancements):
                        enhanced_count += 1
                    else:
                        failed_count += 1
        
        print(f"Enhanced {enhanced_count} images, {failed_count} failed")
        return enhanced_count, failed_count, self.enhancement_stats

def calculate_quality_improvement(original_dir, enhanced_dir):
    """Calculate quality improvement metrics"""
    from pre_processing.modules.image_processing import ImageDatasetCleaner
    
    # Analyze original dataset
    original_cleaner = ImageDatasetCleaner(original_dir)
    original_df = original_cleaner.analyze_dataset()
    
    # Analyze enhanced dataset
    enhanced_cleaner = ImageDatasetCleaner(enhanced_dir)
    enhanced_df = enhanced_cleaner.analyze_dataset()
    
    if original_df.empty or enhanced_df.empty:
        return None
    
    # Calculate improvements
    improvements = {
        'avg_brightness_change': float(enhanced_df['brightness'].mean() - original_df['brightness'].mean()),
        'avg_contrast_change': float(enhanced_df['contrast'].mean() - original_df['contrast'].mean()),
        'avg_blur_change': float(enhanced_df['blur_score'].mean() - original_df['blur_score'].mean()),
        'brightness_std_reduction': float(original_df['brightness'].std() - enhanced_df['brightness'].std()),
        'contrast_improvement': float((enhanced_df['contrast'] > original_df['contrast']).mean() * 100),
        'sharpness_improvement': float((enhanced_df['blur_score'] > original_df['blur_score']).mean() * 100)
    }
    
    return improvements