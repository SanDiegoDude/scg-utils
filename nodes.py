import torch
import numpy as np
import math
import re
from PIL import Image

class SCGZeroedOutputs:
    """
    A utility node that provides zeroed/empty outputs for all common ComfyUI data types.
    Useful as placeholder inputs when building workflows.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {}
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("image", "string", "int", "float", "boolean")
    FUNCTION = "get_zeroed_outputs"
    CATEGORY = "scg-utils"
    
    def get_zeroed_outputs(self):
        # Create a 1x1 black pixel image tensor in ComfyUI format
        # Format: (batch, height, width, channels) with values 0-1
        black_pixel = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        
        # Return all zeroed outputs
        return (
            black_pixel,  # IMAGE: 1x1 black pixel
            "",           # STRING: empty string
            0,            # INT: zero
            0.0,          # FLOAT: zero float
            False         # BOOLEAN: false
        )


class SCGImageStack:
    """
    A utility node that stacks multiple images into a grid layout.
    Accepts 0-4 optional image inputs and arranges them in a customizable grid.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "pad_to_match": ("BOOLEAN", {"default": True}),
                "pad_first_to_square": ("BOOLEAN", {"default": False}),
                "padding_color": (["black", "white"], {"default": "black"}),
                "grid_size": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "rescale_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ignore_duplicates": ("BOOLEAN", {"default": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stack_images"
    CATEGORY = "scg-utils"
    
    def pad_image_to_square(self, image, color_value):
        """Pad image to square dimensions"""
        batch, height, width, channels = image.shape
        max_dim = max(height, width)
        
        if height == width:
            return image
            
        # Calculate padding
        pad_h = max_dim - height
        pad_w = max_dim - width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Create padded image
        padded = torch.full((batch, max_dim, max_dim, channels), color_value, dtype=image.dtype)
        padded[:, pad_top:pad_top+height, pad_left:pad_left+width, :] = image
        
        return padded
    
    def resize_image_to_height(self, image, target_height):
        """Resize image to match target height while maintaining aspect ratio"""
        batch, height, width, channels = image.shape
        
        if height == target_height:
            return image
            
        # Calculate new width maintaining aspect ratio
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        # Resize using torch interpolate
        # Convert from BHWC to BCHW for interpolation
        image_bchw = image.permute(0, 3, 1, 2)
        resized_bchw = torch.nn.functional.interpolate(
            image_bchw, 
            size=(target_height, new_width), 
            mode='bilinear', 
            align_corners=False
        )
        # Convert back to BHWC
        resized = resized_bchw.permute(0, 2, 3, 1)
        
        return resized
    
    def resize_image_to_width(self, image, target_width):
        """Resize image to match target width while maintaining aspect ratio"""
        batch, height, width, channels = image.shape
        
        if width == target_width:
            return image
            
        # Calculate new height maintaining aspect ratio
        aspect_ratio = height / width
        new_height = int(target_width * aspect_ratio)
        
        # Resize using torch interpolate
        # Convert from BHWC to BCHW for interpolation
        image_bchw = image.permute(0, 3, 1, 2)
        resized_bchw = torch.nn.functional.interpolate(
            image_bchw, 
            size=(new_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
        # Convert back to BHWC
        resized = resized_bchw.permute(0, 2, 3, 1)
        
        return resized
    
    def pad_image_to_width(self, image, target_width, color_value):
        """Pad image to target width"""
        batch, height, width, channels = image.shape
        
        if width >= target_width:
            return image
            
        pad_total = target_width - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        padded = torch.full((batch, height, target_width, channels), color_value, dtype=image.dtype)
        padded[:, :, pad_left:pad_left+width, :] = image
        
        return padded
    
    def fit_and_pad_image(self, image, target_height, target_width, color_value):
        """Resize image to fit within target dimensions, then pad to exact size"""
        batch, height, width, channels = image.shape
        
        # Calculate scale factor to fit within target dimensions
        scale_h = target_height / height
        scale_w = target_width / width
        scale = min(scale_h, scale_w)  # Use smaller scale to fit entirely within
        
        # Calculate new dimensions
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Resize image
        if new_height != height or new_width != width:
            # Convert from BHWC to BCHW for interpolation
            image_bchw = image.permute(0, 3, 1, 2)
            resized_bchw = torch.nn.functional.interpolate(
                image_bchw, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )
            # Convert back to BHWC
            resized = resized_bchw.permute(0, 2, 3, 1)
        else:
            resized = image
        
        # If already the right size, return as-is
        if new_height == target_height and new_width == target_width:
            return resized
        
        # Pad to target dimensions
        pad_h = target_height - new_height
        pad_w = target_width - new_width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Create padded image
        padded = torch.full((batch, target_height, target_width, channels), color_value, dtype=image.dtype)
        padded[:, pad_top:pad_top+new_height, pad_left:pad_left+new_width, :] = resized
        
        return padded
    
    def resize_image_largest_side(self, image, target_height, target_width):
        """Resize image using the largest side to maintain aspect ratio better"""
        batch, height, width, channels = image.shape
        
        # Calculate scale factors
        scale_h = target_height / height
        scale_w = target_width / width
        
        # Use the larger scale to resize on the largest side
        # This prevents unnecessary cropping when aspect ratios are similar
        scale = max(scale_h, scale_w)
        
        # Calculate new dimensions
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Resize image
        if new_height != height or new_width != width:
            # Convert from BHWC to BCHW for interpolation
            image_bchw = image.permute(0, 3, 1, 2)
            resized_bchw = torch.nn.functional.interpolate(
                image_bchw, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )
            # Convert back to BHWC
            resized = resized_bchw.permute(0, 2, 3, 1)
        else:
            resized = image
        
        return resized
    
    def create_empty_image(self, height, width, color_value):
        """Create an empty image with specified dimensions and color"""
        return torch.full((1, height, width, 3), color_value, dtype=torch.float32)
    
    def images_are_identical(self, img1, img2):
        """Check if two images are identical (exact pixel match)"""
        if img1.shape != img2.shape:
            return False
        return torch.allclose(img1, img2, rtol=0, atol=0)
    
    def remove_duplicate_images(self, images):
        """Remove duplicate images from the list, keeping only unique images"""
        if not images:
            return images
        
        unique_images = [images[0]]  # Keep the first image
        
        for img in images[1:]:
            is_duplicate = False
            for unique_img in unique_images:
                if self.images_are_identical(img, unique_img):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_images.append(img)
        
        return unique_images
    
    def stack_images(self, rows=2, columns=2, pad_to_match=True, pad_first_to_square=False, 
                    padding_color="black", grid_size=0, rescale_output=False, 
                    ignore_duplicates=False, image1=None, image2=None, image3=None, image4=None):
        
        # Collect non-None images
        images = []
        for img in [image1, image2, image3, image4]:
            if img is not None:
                images.append(img)
        
        # Remove duplicates if requested
        if ignore_duplicates and len(images) > 1:
            images = self.remove_duplicate_images(images)
        
        # If no images, create a 1x1 black pixel (like SCGZeroedOutputs)
        if not images:
            black_pixel = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return (black_pixel,)
        
        # Color value for padding
        color_value = 0.0 if padding_color == "black" else 1.0
        
        # Process first image (this is our reference)
        first_image = images[0]
        
        # Apply pad_first_to_square BEFORE size limiting to avoid redundant processing
        if pad_first_to_square:
            first_image = self.pad_image_to_square(first_image, color_value)
        
        # Limit first image to max 1.24 MP to keep output size manageable
        MAX_PIXELS = 1_240_000  # 1.24 megapixels
        batch, height, width, channels = first_image.shape
        current_pixels = height * width
        
        if current_pixels > MAX_PIXELS:
            # Calculate scale factor to reduce to exactly 1.24 MP
            scale_factor = (MAX_PIXELS / current_pixels) ** 0.5
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            # Resize first image
            image_bchw = first_image.permute(0, 3, 1, 2)
            resized_bchw = torch.nn.functional.interpolate(
                image_bchw, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )
            first_image = resized_bchw.permute(0, 2, 3, 1)
        
        # If only one image, set it as final result but still apply rescaling if needed
        if len(images) == 1:
            final_image = first_image
        else:
            # Multiple images - create grid
            target_height = first_image.shape[1]
            reference_width = first_image.shape[2]  # First image width is the reference
            
            # Process all images based on padding settings
            processed_images = [first_image]
            
            if pad_to_match:
                # When pad_to_match is True: resize images to fit within target dimensions, then pad to exact size
                for img in images[1:]:
                    # Use fit_and_pad_image which handles both resizing and padding correctly
                    img = self.fit_and_pad_image(img, target_height, reference_width, color_value)
                    processed_images.append(img)
            else:
                # When pad_to_match is False: just add images as-is for now
                for img in images[1:]:
                    processed_images.append(img)
            
            # Calculate total slots needed
            total_slots = rows * columns
            
            # Add empty images for missing slots
            while len(processed_images) < total_slots:
                if len(processed_images) > 0:
                    # Use dimensions from first processed image
                    h, w = processed_images[0].shape[1], processed_images[0].shape[2]
                    empty_img = self.create_empty_image(h, w, color_value)
                    processed_images.append(empty_img)
                else:
                    break
            

            
            # Create grid
            grid_rows = []
            for row in range(rows):
                row_images = []
                for col in range(columns):
                    idx = row * columns + col
                    if idx < len(processed_images):
                        img = processed_images[idx]
                        # Add grid spacing if specified
                        if grid_size > 0:
                            h, w = img.shape[1], img.shape[2]
                            padded_img = torch.full(
                                (1, h + 2*grid_size, w + 2*grid_size, 3), 
                                color_value, 
                                dtype=img.dtype
                            )
                            padded_img[:, grid_size:grid_size+h, grid_size:grid_size+w, :] = img
                            row_images.append(padded_img)
                        else:
                            row_images.append(img)
                
                if row_images:
                    # Ensure all images in the row have the same height before concatenation
                    # This is crucial for rescale_output to work properly
                    # Only needed when pad_to_match=False, since fit_and_pad_image already ensures same dimensions
                    if not pad_to_match and len(row_images) > 1:
                        # Find the target height (use the height of the first image in the row)
                        target_height = row_images[0].shape[1]
                        
                        # Normalize all images in the row to the same height
                        for i in range(len(row_images)):
                            if row_images[i].shape[1] != target_height:
                                row_images[i] = self.resize_image_to_height(row_images[i], target_height)
                    
                    # Concatenate images horizontally
                    row_concat = torch.cat(row_images, dim=2)
                    grid_rows.append(row_concat)
            
            # When pad_to_match is False, ensure all rows have same dimensions for vertical concatenation
            # Use the first row's dimensions as reference (consistent with height normalization)
            if not pad_to_match and len(grid_rows) > 1:
                # Use the first row's dimensions as the target
                target_height = grid_rows[0].shape[1]
                target_width = grid_rows[0].shape[2]
                
                # Normalize each row to target dimensions
                for i, row in enumerate(grid_rows):
                    batch, height, width, channels = row.shape
                    
                    # Handle height differences
                    if height != target_height:
                        if height < target_height:
                            # Pad to target height
                            pad_h = target_height - height
                            pad_top = pad_h // 2
                            pad_bottom = pad_h - pad_top
                            padded_row = torch.full((batch, target_height, width, channels), color_value, dtype=row.dtype)
                            padded_row[:, pad_top:pad_top+height, :, :] = row
                            row = padded_row
                        else:
                            # Crop to target height (shouldn't happen with current logic, but just in case)
                            row = row[:, :target_height, :, :]
                    
                    # Handle width differences
                    if row.shape[2] != target_width:
                        if row.shape[2] < target_width:
                            # Pad to target width
                            current_width = row.shape[2]
                            batch, height, width, channels = row.shape
                            padded_row = torch.full((batch, height, target_width, channels), color_value, dtype=row.dtype)
                            padded_row[:, :, :current_width, :] = row
                            grid_rows[i] = padded_row
                        else:
                            # Crop to target width (shouldn't happen with current logic, but just in case)
                            grid_rows[i] = row[:, :, :target_width, :]
                    else:
                        grid_rows[i] = row
            
            # Concatenate rows vertically
            if grid_rows:
                final_image = torch.cat(grid_rows, dim=1)
            else:
                final_image = first_image
        
        # Apply output rescaling if requested (works for both single images and grids)
        if rescale_output:
            MAX_OUTPUT_PIXELS = 1_240_000  # 1.24 megapixels
            batch, height, width, channels = final_image.shape
            current_pixels = height * width
            
            if current_pixels > MAX_OUTPUT_PIXELS:
                # Calculate scale factor to reduce to exactly 1.24 MP
                scale_factor = (MAX_OUTPUT_PIXELS / current_pixels) ** 0.5
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                
                # Resize final output
                image_bchw = final_image.permute(0, 3, 1, 2)
                resized_bchw = torch.nn.functional.interpolate(
                    image_bchw, 
                    size=(new_height, new_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                final_image = resized_bchw.permute(0, 2, 3, 1)
        
        return (final_image,)
        
        # Fallback - should never reach here
        black_pixel = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return (black_pixel,)


class SCGImageStackXL:
    """
    A utility node that stacks multiple images into a grid layout.
    Accepts 0-8 optional image inputs and arranges them in a customizable grid.
    Extended version of SCGImageStack with support for more images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "pad_to_match": ("BOOLEAN", {"default": True}),
                "pad_first_to_square": ("BOOLEAN", {"default": False}),
                "padding_color": (["black", "white"], {"default": "black"}),
                "grid_size": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "rescale_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ignore_duplicates": ("BOOLEAN", {"default": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stack_images"
    CATEGORY = "scg-utils"
    
    def pad_image_to_square(self, image, color_value):
        """Pad image to square dimensions"""
        batch, height, width, channels = image.shape
        max_dim = max(height, width)
        
        if height == width:
            return image
            
        # Calculate padding
        pad_h = max_dim - height
        pad_w = max_dim - width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Create padded image
        padded = torch.full((batch, max_dim, max_dim, channels), color_value, dtype=image.dtype)
        padded[:, pad_top:pad_top+height, pad_left:pad_left+width, :] = image
        
        return padded
    
    def resize_image_to_height(self, image, target_height):
        """Resize image to match target height while maintaining aspect ratio"""
        batch, height, width, channels = image.shape
        
        if height == target_height:
            return image
            
        # Calculate new width maintaining aspect ratio
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        # Resize using torch interpolate
        # Convert from BHWC to BCHW for interpolation
        image_bchw = image.permute(0, 3, 1, 2)
        resized_bchw = torch.nn.functional.interpolate(
            image_bchw, 
            size=(target_height, new_width), 
            mode='bilinear', 
            align_corners=False
        )
        # Convert back to BHWC
        resized = resized_bchw.permute(0, 2, 3, 1)
        
        return resized
    
    def resize_image_to_width(self, image, target_width):
        """Resize image to match target width while maintaining aspect ratio"""
        batch, height, width, channels = image.shape
        
        if width == target_width:
            return image
            
        # Calculate new height maintaining aspect ratio
        aspect_ratio = height / width
        new_height = int(target_width * aspect_ratio)
        
        # Resize using torch interpolate
        # Convert from BHWC to BCHW for interpolation
        image_bchw = image.permute(0, 3, 1, 2)
        resized_bchw = torch.nn.functional.interpolate(
            image_bchw, 
            size=(new_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
        # Convert back to BHWC
        resized = resized_bchw.permute(0, 2, 3, 1)
        
        return resized
    
    def pad_image_to_width(self, image, target_width, color_value):
        """Pad image to target width"""
        batch, height, width, channels = image.shape
        
        if width >= target_width:
            return image
            
        pad_total = target_width - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        padded = torch.full((batch, height, target_width, channels), color_value, dtype=image.dtype)
        padded[:, :, pad_left:pad_left+width, :] = image
        
        return padded
    
    def fit_and_pad_image(self, image, target_height, target_width, color_value):
        """Resize image to fit within target dimensions, then pad to exact size"""
        batch, height, width, channels = image.shape
        
        # Calculate scale factor to fit within target dimensions
        scale_h = target_height / height
        scale_w = target_width / width
        scale = min(scale_h, scale_w)  # Use smaller scale to fit entirely within
        
        # Calculate new dimensions
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Resize image
        if new_height != height or new_width != width:
            # Convert from BHWC to BCHW for interpolation
            image_bchw = image.permute(0, 3, 1, 2)
            resized_bchw = torch.nn.functional.interpolate(
                image_bchw, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )
            # Convert back to BHWC
            resized = resized_bchw.permute(0, 2, 3, 1)
        else:
            resized = image
        
        # If already the right size, return as-is
        if new_height == target_height and new_width == target_width:
            return resized
        
        # Pad to target dimensions
        pad_h = target_height - new_height
        pad_w = target_width - new_width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Create padded image
        padded = torch.full((batch, target_height, target_width, channels), color_value, dtype=image.dtype)
        padded[:, pad_top:pad_top+new_height, pad_left:pad_left+new_width, :] = resized
        
        return padded
    
    def resize_image_largest_side(self, image, target_height, target_width):
        """Resize image using the largest side to maintain aspect ratio better"""
        batch, height, width, channels = image.shape
        
        # Calculate scale factors
        scale_h = target_height / height
        scale_w = target_width / width
        
        # Use the larger scale to resize on the largest side
        # This prevents unnecessary cropping when aspect ratios are similar
        scale = max(scale_h, scale_w)
        
        # Calculate new dimensions
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Resize image
        if new_height != height or new_width != width:
            # Convert from BHWC to BCHW for interpolation
            image_bchw = image.permute(0, 3, 1, 2)
            resized_bchw = torch.nn.functional.interpolate(
                image_bchw, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )
            # Convert back to BHWC
            resized = resized_bchw.permute(0, 2, 3, 1)
        else:
            resized = image
        
        return resized
    
    def create_empty_image(self, height, width, color_value):
        """Create an empty image with specified dimensions and color"""
        return torch.full((1, height, width, 3), color_value, dtype=torch.float32)
    
    def images_are_identical(self, img1, img2):
        """Check if two images are identical (exact pixel match)"""
        if img1.shape != img2.shape:
            return False
        return torch.allclose(img1, img2, rtol=0, atol=0)
    
    def remove_duplicate_images(self, images):
        """Remove duplicate images from the list, keeping only unique images"""
        if not images:
            return images
        
        unique_images = [images[0]]  # Keep the first image
        
        for img in images[1:]:
            is_duplicate = False
            for unique_img in unique_images:
                if self.images_are_identical(img, unique_img):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_images.append(img)
        
        return unique_images
    
    def stack_images(self, rows=2, columns=2, pad_to_match=True, pad_first_to_square=False, 
                    padding_color="black", grid_size=0, rescale_output=False, 
                    ignore_duplicates=False, image1=None, image2=None, image3=None, image4=None,
                    image5=None, image6=None, image7=None, image8=None):
        
        # Collect non-None images
        images = []
        for img in [image1, image2, image3, image4, image5, image6, image7, image8]:
            if img is not None:
                images.append(img)
        
        # Remove duplicates if requested
        if ignore_duplicates and len(images) > 1:
            images = self.remove_duplicate_images(images)
        
        # If no images, create a 1x1 black pixel (like SCGZeroedOutputs)
        if not images:
            black_pixel = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return (black_pixel,)
        
        # Color value for padding
        color_value = 0.0 if padding_color == "black" else 1.0
        
        # Process first image (this is our reference)
        first_image = images[0]
        
        # Apply pad_first_to_square BEFORE size limiting to avoid redundant processing
        if pad_first_to_square:
            first_image = self.pad_image_to_square(first_image, color_value)
        
        # Limit first image to max 1.24 MP to keep output size manageable
        MAX_PIXELS = 1_240_000  # 1.24 megapixels
        batch, height, width, channels = first_image.shape
        current_pixels = height * width
        
        if current_pixels > MAX_PIXELS:
            # Calculate scale factor to reduce to exactly 1.24 MP
            scale_factor = (MAX_PIXELS / current_pixels) ** 0.5
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            # Resize first image
            image_bchw = first_image.permute(0, 3, 1, 2)
            resized_bchw = torch.nn.functional.interpolate(
                image_bchw, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )
            first_image = resized_bchw.permute(0, 2, 3, 1)
        
        # If only one image, set it as final result but still apply rescaling if needed
        if len(images) == 1:
            final_image = first_image
        else:
            # Multiple images - create grid
            target_height = first_image.shape[1]
            reference_width = first_image.shape[2]  # First image width is the reference
            
            # Process all images based on padding settings
            processed_images = [first_image]
            
            if pad_to_match:
                # When pad_to_match is True: resize images to fit within target dimensions, then pad to exact size
                for img in images[1:]:
                    # Use fit_and_pad_image which handles both resizing and padding correctly
                    img = self.fit_and_pad_image(img, target_height, reference_width, color_value)
                    processed_images.append(img)
            else:
                # When pad_to_match is False: just add images as-is for now
                for img in images[1:]:
                    processed_images.append(img)
            
            # Calculate total slots needed
            total_slots = rows * columns
            
            # Add empty images for missing slots
            while len(processed_images) < total_slots:
                if len(processed_images) > 0:
                    # Use dimensions from first processed image
                    h, w = processed_images[0].shape[1], processed_images[0].shape[2]
                    empty_img = self.create_empty_image(h, w, color_value)
                    processed_images.append(empty_img)
                else:
                    break
            

            
            # Create grid
            grid_rows = []
            for row in range(rows):
                row_images = []
                for col in range(columns):
                    idx = row * columns + col
                    if idx < len(processed_images):
                        img = processed_images[idx]
                        # Add grid spacing if specified
                        if grid_size > 0:
                            h, w = img.shape[1], img.shape[2]
                            padded_img = torch.full(
                                (1, h + 2*grid_size, w + 2*grid_size, 3), 
                                color_value, 
                                dtype=img.dtype
                            )
                            padded_img[:, grid_size:grid_size+h, grid_size:grid_size+w, :] = img
                            row_images.append(padded_img)
                        else:
                            row_images.append(img)
                
                if row_images:
                    # Ensure all images in the row have the same height before concatenation
                    # This is crucial for rescale_output to work properly
                    # Only needed when pad_to_match=False, since fit_and_pad_image already ensures same dimensions
                    if not pad_to_match and len(row_images) > 1:
                        # Find the target height (use the height of the first image in the row)
                        target_height = row_images[0].shape[1]
                        
                        # Normalize all images in the row to the same height
                        for i in range(len(row_images)):
                            if row_images[i].shape[1] != target_height:
                                row_images[i] = self.resize_image_to_height(row_images[i], target_height)
                    
                    # Concatenate images horizontally
                    row_concat = torch.cat(row_images, dim=2)
                    grid_rows.append(row_concat)
            
            # When pad_to_match is False, ensure all rows have same dimensions for vertical concatenation
            # Use the first row's dimensions as reference (consistent with height normalization)
            if not pad_to_match and len(grid_rows) > 1:
                # Use the first row's dimensions as the target
                target_height = grid_rows[0].shape[1]
                target_width = grid_rows[0].shape[2]
                
                # Normalize each row to target dimensions
                for i, row in enumerate(grid_rows):
                    batch, height, width, channels = row.shape
                    
                    # Handle height differences
                    if height != target_height:
                        if height < target_height:
                            # Pad to target height
                            pad_h = target_height - height
                            pad_top = pad_h // 2
                            pad_bottom = pad_h - pad_top
                            padded_row = torch.full((batch, target_height, width, channels), color_value, dtype=row.dtype)
                            padded_row[:, pad_top:pad_top+height, :, :] = row
                            row = padded_row
                        else:
                            # Crop to target height (shouldn't happen with current logic, but just in case)
                            row = row[:, :target_height, :, :]
                    
                    # Handle width differences
                    if row.shape[2] != target_width:
                        if row.shape[2] < target_width:
                            # Pad to target width
                            current_width = row.shape[2]
                            batch, height, width, channels = row.shape
                            padded_row = torch.full((batch, height, target_width, channels), color_value, dtype=row.dtype)
                            padded_row[:, :, :current_width, :] = row
                            grid_rows[i] = padded_row
                        else:
                            # Crop to target width (shouldn't happen with current logic, but just in case)
                            grid_rows[i] = row[:, :, :target_width, :]
                    else:
                        grid_rows[i] = row
            
            # Concatenate rows vertically
            if grid_rows:
                final_image = torch.cat(grid_rows, dim=1)
            else:
                final_image = first_image
        
        # Apply output rescaling if requested (works for both single images and grids)
        if rescale_output:
            MAX_OUTPUT_PIXELS = 1_240_000  # 1.24 megapixels
            batch, height, width, channels = final_image.shape
            current_pixels = height * width
            
            if current_pixels > MAX_OUTPUT_PIXELS:
                # Calculate scale factor to reduce to exactly 1.24 MP
                scale_factor = (MAX_OUTPUT_PIXELS / current_pixels) ** 0.5
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                
                # Resize final output
                image_bchw = final_image.permute(0, 3, 1, 2)
                resized_bchw = torch.nn.functional.interpolate(
                    image_bchw, 
                    size=(new_height, new_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                final_image = resized_bchw.permute(0, 2, 3, 1)
        
        return (final_image,)
        
        # Fallback - should never reach here
        black_pixel = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return (black_pixel,)


class SCGWildcardVariableProcessor:
    """
    A utility node that replaces wildcard variables in text strings.
    Useful for injecting dynamic content into prompts and system messages.
    
    Replaces {wildcard_name} in the input string with the provided replacement text.
    Sanitizes the replacement text to prevent breaking LLM API calls.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter text with {wildcards}"}),
                "wildcard_name": ("STRING", {"default": "", "placeholder": "Variable name (without braces)"}),
                "replacement_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Text to replace the wildcard"}),
                "sanitize_text": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_string",)
    FUNCTION = "process_wildcards"
    CATEGORY = "scg-utils"
    
    def sanitize_replacement(self, text):
        """
        Sanitize text to prevent breaking LLM API calls.
        Escapes problematic characters while keeping text readable.
        """
        if not text:
            return text
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize quotes to prevent JSON breaking
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Escape backslashes that aren't already escaped
        text = re.sub(r'(?<!\\)\\(?!["\\nrt])', r'\\\\', text)
        
        # Remove or replace other control characters (except newlines and tabs)
        # Keep \n and \t as they're often intentional in prompts
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        
        return text
    
    def process_wildcards(self, input_string, wildcard_name, replacement_text, sanitize_text):
        """
        Replace wildcard variables in the input string.
        
        Args:
            input_string: Template string containing {wildcard_name} placeholders
            wildcard_name: Name of the variable to replace (without braces)
            replacement_text: Text to replace the wildcard with
            sanitize_text: Whether to sanitize the replacement text
        
        Returns:
            Tuple containing the processed string
        """
        if not input_string:
            return ("",)
        
        if not wildcard_name:
            # No wildcard name provided, return original string
            return (input_string,)
        
        # Sanitize replacement text if requested
        if sanitize_text:
            processed_replacement = self.sanitize_replacement(replacement_text)
        else:
            processed_replacement = replacement_text
        
        # Build the wildcard pattern: {wildcard_name}
        wildcard_pattern = f"{{{wildcard_name}}}"
        
        # Replace all occurrences of the wildcard
        processed_string = input_string.replace(wildcard_pattern, processed_replacement)
        
        # Log the replacement for debugging
        replacement_count = input_string.count(wildcard_pattern)
        if replacement_count > 0:
            print(f"[SCG Wildcard] Replaced {replacement_count} occurrence(s) of '{wildcard_pattern}'")
        else:
            print(f"[SCG Wildcard] Warning: No occurrences of '{wildcard_pattern}' found in input string")
        
        return (processed_string,)


class SCGScaleToMegapixels:
    """
    A utility node that scales images to a specific megapixel size.
    Maintains aspect ratio while scaling to the target megapixel count.
    Supports multiple interpolation methods for high-quality results.
    Optionally constrains dimensions to be divisible by a specified value.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 16.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "scaling_method": ([
                    "lanczos",
                    "bicubic", 
                    "bilinear",
                    "nearest",
                    "area"
                ], {
                    "default": "lanczos"
                }),
                "dimension_constraint": ([
                    "resize",
                    "crop",
                    "none"
                ], {
                    "default": "resize"
                }),
                "divisible_by": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_to_megapixels"
    CATEGORY = "scg-utils"
    
    def _apply_dimension_constraint(self, width, height, divisible_by, constraint_mode, scaling_method, image=None):
        """
        Apply dimension constraint (resize or crop) to make dimensions divisible.
        
        Args:
            width: Current width
            height: Current height
            divisible_by: Value dimensions should be divisible by
            constraint_mode: "resize", "crop", or "none"
            scaling_method: Method to use if resizing
            image: Image tensor (required for crop/resize operations)
        
        Returns:
            Tuple of (new_width, new_height, modified_image or None)
        """
        if constraint_mode == "none" or divisible_by <= 1:
            return width, height, None
        
        # Calculate target dimensions (round down to nearest divisible)
        target_width = (width // divisible_by) * divisible_by
        target_height = (height // divisible_by) * divisible_by
        
        # Ensure minimum dimensions
        target_width = max(divisible_by, target_width)
        target_height = max(divisible_by, target_height)
        
        if target_width == width and target_height == height:
            return width, height, None
        
        if constraint_mode == "crop" and image is not None:
            # Center crop to target dimensions
            crop_x = (width - target_width) // 2
            crop_y = (height - target_height) // 2
            cropped = image[:, crop_y:crop_y + target_height, crop_x:crop_x + target_width, :]
            return target_width, target_height, cropped
        elif constraint_mode == "resize" and image is not None:
            # Resize to target dimensions
            resized = self._scale_image(image, target_height, target_width, scaling_method)
            return target_width, target_height, resized
        
        return target_width, target_height, None
    
    def _scale_image(self, image, new_height, new_width, scaling_method):
        """Scale image using the specified method."""
        if scaling_method == "lanczos":
            return self._scale_with_pil(image, new_height, new_width, Image.LANCZOS)
        elif scaling_method == "area":
            return self._scale_with_pil(image, new_height, new_width, Image.BOX)
        else:
            return self._scale_with_torch(image, new_height, new_width, scaling_method)
    
    def scale_to_megapixels(self, image, megapixels, scaling_method, dimension_constraint, divisible_by):
        """
        Scale image to target megapixel size while maintaining aspect ratio.
        
        Args:
            image: Input image tensor in ComfyUI format (batch, height, width, channels)
            megapixels: Target size in megapixels (e.g., 1.0 = 1,000,000 pixels)
            scaling_method: Interpolation method to use
            dimension_constraint: How to handle non-divisible dimensions ("resize", "crop", "none")
            divisible_by: Value that dimensions should be divisible by
        
        Returns:
            Tuple containing the scaled image, width, and height
        """
        batch, height, width, channels = image.shape
        current_pixels = height * width
        target_pixels = int(megapixels * 1_000_000)
        
        # Calculate scaling factor
        scale_factor = (target_pixels / current_pixels) ** 0.5
        
        # Calculate new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Ensure dimensions are at least 1
        new_height = max(1, new_height)
        new_width = max(1, new_width)
        
        batch_info = f" (batch of {batch} images)" if batch > 1 else ""
        print(f"[SCG Scale to Megapixels] Scaling{batch_info} from {width}x{height} ({current_pixels/1_000_000:.2f}MP) to {new_width}x{new_height} ({(new_width*new_height)/1_000_000:.2f}MP) using {scaling_method}")
        
        # Scale image if dimensions changed
        if new_height == height and new_width == width:
            scaled_image = image
        else:
            scaled_image = self._scale_image(image, new_height, new_width, scaling_method)
        
        # Apply dimension constraint if needed
        final_width, final_height, constrained_image = self._apply_dimension_constraint(
            new_width, new_height, divisible_by, dimension_constraint, scaling_method, scaled_image
        )
        
        if constrained_image is not None:
            scaled_image = constrained_image
            if final_width != new_width or final_height != new_height:
                print(f"[SCG Scale to Megapixels] Applied {dimension_constraint} constraint: {new_width}x{new_height} -> {final_width}x{final_height} (divisible by {divisible_by})")
        else:
            final_width, final_height = new_width, new_height
        
        return (scaled_image, final_width, final_height)
    
    def _scale_with_torch(self, image, new_height, new_width, method):
        """
        Scale image using PyTorch interpolation.
        
        Args:
            image: Input image tensor (batch, height, width, channels)
            new_height: Target height
            new_width: Target width
            method: Interpolation method ('bicubic', 'bilinear', 'nearest')
        
        Returns:
            Scaled image tensor
        """
        # Convert from BHWC to BCHW for interpolation
        image_bchw = image.permute(0, 3, 1, 2)
        
        # Perform interpolation
        resized_bchw = torch.nn.functional.interpolate(
            image_bchw,
            size=(new_height, new_width),
            mode=method,
            align_corners=False if method != 'nearest' else None,
            antialias=True if method in ['bicubic', 'bilinear'] else False
        )
        
        # Convert back to BHWC
        resized = resized_bchw.permute(0, 2, 3, 1)
        
        return resized
    
    def _scale_with_pil(self, image, new_height, new_width, resample_method):
        """
        Scale image using PIL for methods not available in PyTorch.
        
        Args:
            image: Input image tensor (batch, height, width, channels)
            new_height: Target height
            new_width: Target width
            resample_method: PIL resampling method (e.g., Image.LANCZOS)
        
        Returns:
            Scaled image tensor
        """
        batch, height, width, channels = image.shape
        
        # Process each image in the batch
        scaled_images = []
        for i in range(batch):
            # Convert tensor to PIL Image
            # Tensor is in range [0, 1], convert to [0, 255] for PIL
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Resize using PIL
            pil_img_resized = pil_img.resize((new_width, new_height), resample=resample_method)
            
            # Convert back to tensor
            img_np_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np_resized)
            
            scaled_images.append(img_tensor)
        
        # Stack back into batch
        scaled_batch = torch.stack(scaled_images, dim=0)
        
        return scaled_batch


class SCGScaleDimensionToSize:
    """
    A utility node that scales images by targeting a specific dimension size.
    Can target the shortest side, longest side, width, or height.
    Maintains aspect ratio while scaling to the target dimension.
    Supports multiple interpolation methods for high-quality results.
    Optionally constrains dimensions to be divisible by a specified value.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "step": 1
                }),
                "apply_to": ([
                    "shortest",
                    "longest",
                    "width",
                    "height"
                ], {
                    "default": "shortest"
                }),
                "scaling_method": ([
                    "lanczos",
                    "bicubic", 
                    "bilinear",
                    "nearest",
                    "area"
                ], {
                    "default": "lanczos"
                }),
                "dimension_constraint": ([
                    "resize",
                    "crop",
                    "none"
                ], {
                    "default": "resize"
                }),
                "divisible_by": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_to_dimension"
    CATEGORY = "scg-utils"
    
    def _apply_dimension_constraint(self, width, height, divisible_by, constraint_mode, scaling_method, image=None):
        """
        Apply dimension constraint (resize or crop) to make dimensions divisible.
        """
        if constraint_mode == "none" or divisible_by <= 1:
            return width, height, None
        
        # Calculate target dimensions (round down to nearest divisible)
        target_width = (width // divisible_by) * divisible_by
        target_height = (height // divisible_by) * divisible_by
        
        # Ensure minimum dimensions
        target_width = max(divisible_by, target_width)
        target_height = max(divisible_by, target_height)
        
        if target_width == width and target_height == height:
            return width, height, None
        
        if constraint_mode == "crop" and image is not None:
            # Center crop to target dimensions
            crop_x = (width - target_width) // 2
            crop_y = (height - target_height) // 2
            cropped = image[:, crop_y:crop_y + target_height, crop_x:crop_x + target_width, :]
            return target_width, target_height, cropped
        elif constraint_mode == "resize" and image is not None:
            # Resize to target dimensions
            resized = self._scale_image(image, target_height, target_width, scaling_method)
            return target_width, target_height, resized
        
        return target_width, target_height, None
    
    def _scale_image(self, image, new_height, new_width, scaling_method):
        """Scale image using the specified method."""
        if scaling_method == "lanczos":
            return self._scale_with_pil(image, new_height, new_width, Image.LANCZOS)
        elif scaling_method == "area":
            return self._scale_with_pil(image, new_height, new_width, Image.BOX)
        else:
            return self._scale_with_torch(image, new_height, new_width, scaling_method)
    
    def scale_to_dimension(self, image, target_size, apply_to, scaling_method, dimension_constraint, divisible_by):
        """
        Scale image by targeting a specific dimension size.
        
        Args:
            image: Input image tensor in ComfyUI format (batch, height, width, channels)
            target_size: Target size for the selected dimension
            apply_to: Which dimension to target ("shortest", "longest", "width", "height")
            scaling_method: Interpolation method to use
            dimension_constraint: How to handle non-divisible dimensions ("resize", "crop", "none")
            divisible_by: Value that dimensions should be divisible by
        
        Returns:
            Tuple containing the scaled image, width, and height
        """
        batch, height, width, channels = image.shape
        
        # Determine which dimension to use for scaling
        if apply_to == "shortest":
            reference_dim = min(width, height)
        elif apply_to == "longest":
            reference_dim = max(width, height)
        elif apply_to == "width":
            reference_dim = width
        else:  # height
            reference_dim = height
        
        # Calculate scale factor
        scale_factor = target_size / reference_dim
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure dimensions are at least 1
        new_height = max(1, new_height)
        new_width = max(1, new_width)
        
        batch_info = f" (batch of {batch} images)" if batch > 1 else ""
        print(f"[SCG Scale Dimension to Size] Scaling{batch_info} from {width}x{height} to {new_width}x{new_height} ({apply_to}={target_size}) using {scaling_method}")
        
        # Scale image if dimensions changed
        if new_height == height and new_width == width:
            scaled_image = image
        else:
            scaled_image = self._scale_image(image, new_height, new_width, scaling_method)
        
        # Apply dimension constraint if needed
        final_width, final_height, constrained_image = self._apply_dimension_constraint(
            new_width, new_height, divisible_by, dimension_constraint, scaling_method, scaled_image
        )
        
        if constrained_image is not None:
            scaled_image = constrained_image
            if final_width != new_width or final_height != new_height:
                print(f"[SCG Scale Dimension to Size] Applied {dimension_constraint} constraint: {new_width}x{new_height} -> {final_width}x{final_height} (divisible by {divisible_by})")
        else:
            final_width, final_height = new_width, new_height
        
        return (scaled_image, final_width, final_height)
    
    def _scale_with_torch(self, image, new_height, new_width, method):
        """Scale image using PyTorch interpolation."""
        image_bchw = image.permute(0, 3, 1, 2)
        
        resized_bchw = torch.nn.functional.interpolate(
            image_bchw,
            size=(new_height, new_width),
            mode=method,
            align_corners=False if method != 'nearest' else None,
            antialias=True if method in ['bicubic', 'bilinear'] else False
        )
        
        resized = resized_bchw.permute(0, 2, 3, 1)
        return resized
    
    def _scale_with_pil(self, image, new_height, new_width, resample_method):
        """Scale image using PIL for methods not available in PyTorch."""
        batch, height, width, channels = image.shape
        
        scaled_images = []
        for i in range(batch):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img_resized = pil_img.resize((new_width, new_height), resample=resample_method)
            img_np_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np_resized)
            scaled_images.append(img_tensor)
        
        scaled_batch = torch.stack(scaled_images, dim=0)
        return scaled_batch


class SCGDimensions:
    """Helper class for SCG Resolution Selector to manage width/height dimensions."""
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value):
        if value < 128:
            raise ValueError("width of less than 128 pixels")
        # Ensure divisible by 16
        self._width = int(value / 16) * 16

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value):
        if value < 128:
            raise ValueError("height of less than 128 pixels")
        # Ensure divisible by 16
        self._height = int(value / 16) * 16


def scg_calculate_aspect_ratio(
    base_resolution: int, ratio: float, overextend: bool
) -> SCGDimensions:
    """Calculate width and height based on base resolution and aspect ratio."""
    width = base_resolution
    height = base_resolution

    if overextend:
        if ratio > 1:
            height *= ratio
        else:
            width /= ratio
    else:
        if ratio > 1:
            width /= ratio
        else:
            height *= ratio

    return SCGDimensions(width, height)


class SCGResolutionSelector:
    """
    A resolution selector node with flexible base resolution control.
    Allows minimum resolution of 128 and base_resolution in increments of 16.
    Calculates width/height based on aspect ratio, ensuring divisibility by 16.
    """
    
    # Default aspect ratios
    DEFAULT_RATIOS = [
        "1:1",
        "landscape (5:4)",
        "landscape (4:3)",
        "landscape (3:2)",
        "landscape (16:10)",
        "landscape (16:9)",
        "landscape (21:9)",
        "portrait (4:5)",
        "portrait (3:4)",
        "portrait (2:3)",
        "portrait (10:9)",
        "portrait (9:16)",
        "portrait (9:21)",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_resolution": (
                    "INT",
                    {
                        "default": 512,
                        "min": 128,
                        "max": 8192,
                        "step": 16,
                    },
                ),
                "aspect_ratio": (cls.DEFAULT_RATIOS,),
                "overextend": (
                    "BOOLEAN",
                    {"default": False, "label_on": "yes", "label_off": "no"},
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    OUTPUT_NODE = False
    CATEGORY = "scg-utils"

    def calculate(
        self,
        base_resolution: int,
        aspect_ratio: str,
        overextend: bool,
    ):
        """Calculate output width and height based on base resolution and aspect ratio."""
        if m := re.search(r"(\d+):(\d+)", aspect_ratio):
            ratio: float = int(m.group(2)) / int(m.group(1))
            d = scg_calculate_aspect_ratio(base_resolution, ratio, overextend)

            # return as dict with `ui` key to trigger onExecuted
            return {
                "ui": {
                    "width": [d.width],
                    "height": [d.height],
                    "ratio": [ratio],
                },
                "result": (d.width, d.height),
            }

        raise ValueError(f"Could't find aspect ratio in string `{aspect_ratio}`")


class SCGTrimImageToMask:
    """
    Trim an image to the masked content, optionally expand context, and return a
    cropped, masked image plus the trimmed mask and restitch data. Useful for
    isolating a subject from a segmentation mask before downstream processing
    while keeping enough metadata to stitch the result back later.
    
    Supports batched mask input from segmentation nodes like SAM3. When multiple
    masks are provided (e.g., from detecting multiple objects), they are combined
    into a single mask using logical OR before processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "trim_image_to_mask": ("BOOLEAN", {"default": True}),
                "isolate_mask": ("BOOLEAN", {"default": True}),
                "resize_output": ("BOOLEAN", {"default": True}),
                "scaling_type": (
                    ["lanczos", "bicubic", "bilinear", "nearest", "area"],
                    {"default": "lanczos"},
                ),
                "resolution": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 16},
                ),
                "mask_expand": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.01},
                ),
                "mask_fill_holes": ("BOOLEAN", {"default": True}),
                "mask_invert": ("BOOLEAN", {"default": False}),
                "context_expand_borders": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.01},
                ),
                "background": (
                    ["black", "white", "transparent"],
                    {"default": "black"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "RESTITCH")
    RETURN_NAMES = ("image", "mask", "restitch_data")
    FUNCTION = "trim_image_to_mask"
    CATEGORY = "scg-utils"

    def _fill_mask_holes(self, mask_np, fill_holes: bool):
        if not fill_holes:
            return mask_np

        try:
            from scipy.ndimage import binary_fill_holes

            return binary_fill_holes(mask_np)
        except Exception as exc:  # pragma: no cover - best effort helper
            print(f"[SCG Trim] binary_fill_holes unavailable, skipping ({exc})")
            return mask_np

    def _dilate_mask(self, mask_tensor, factor: float, bbox_width: int, bbox_height: int):
        """Dilate the binary mask using max pooling to avoid extra deps."""
        if factor <= 1.0:
            return mask_tensor

        # Expansion radius is proportional to the current bbox size
        grow_px = int(max(bbox_width, bbox_height) * (factor - 1.0) / 2)
        if grow_px < 1:
            return mask_tensor

        kernel_size = grow_px * 2 + 1
        pad = grow_px

        mask_batch = mask_tensor.unsqueeze(0).unsqueeze(0)
        dilated = torch.nn.functional.max_pool2d(mask_batch, kernel_size=kernel_size, stride=1, padding=pad)
        return dilated[0, 0]

    def _compute_bbox(self, mask_tensor):
        nonzero = torch.nonzero(mask_tensor > 0.001, as_tuple=False)
        if nonzero.numel() == 0:
            return None

        y_min = int(nonzero[:, 0].min().item())
        x_min = int(nonzero[:, 1].min().item())
        y_max = int(nonzero[:, 0].max().item())
        x_max = int(nonzero[:, 1].max().item())
        return x_min, y_min, x_max, y_max

    def _expand_bbox(self, bbox, expand_factor, img_w, img_h):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        cx = x_min + width / 2.0
        cy = y_min + height / 2.0

        new_w = int(math.ceil(width * expand_factor))
        new_h = int(math.ceil(height * expand_factor))

        new_x_min = max(0, int(round(cx - new_w / 2.0)))
        new_y_min = max(0, int(round(cy - new_h / 2.0)))
        new_x_max = min(img_w - 1, new_x_min + new_w - 1)
        new_y_max = min(img_h - 1, new_y_min + new_h - 1)

        # Adjust if clamped by image edges
        new_w = new_x_max - new_x_min + 1
        new_h = new_y_max - new_y_min + 1

        return new_x_min, new_y_min, new_x_max, new_y_max, new_w, new_h

    def _resize_images(self, images, target_h, target_w, method):
        if method == "lanczos":
            resized = []
            for img in images:
                pil_img = Image.fromarray(
                    (img.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                )
                pil_resized = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)
                resized.append(
                    torch.from_numpy(np.array(pil_resized).astype(np.float32) / 255.0).to(img.device)
                )
            return torch.stack(resized, dim=0)

        bchw = images.permute(0, 3, 1, 2)
        if method in ["bilinear", "bicubic"]:
            resized_bchw = torch.nn.functional.interpolate(
                bchw, size=(target_h, target_w), mode=method, align_corners=False, antialias=True
            )
        elif method == "area":
            resized_bchw = torch.nn.functional.interpolate(
                bchw, size=(target_h, target_w), mode=method
            )
        else:  # nearest or other torch-supported modes without align_corners
            resized_bchw = torch.nn.functional.interpolate(
                bchw, size=(target_h, target_w), mode=method
            )
        return resized_bchw.permute(0, 2, 3, 1)

    def _resize_masks(self, masks, target_h, target_w):
        masks = masks.unsqueeze(1)  # B,1,H,W
        resized = torch.nn.functional.interpolate(masks, size=(target_h, target_w), mode="nearest")
        return resized.squeeze(1)

    def _combine_masks(self, masks):
        """
        Combine multiple masks into a single mask using logical OR.
        
        Args:
            masks: Tensor of shape [N, H, W] where N is number of masks
            
        Returns:
            Combined mask of shape [1, H, W]
        """
        # Use max across the batch dimension to combine masks (logical OR for binary masks)
        combined = torch.max(masks, dim=0, keepdim=True)[0]
        return combined

    def trim_image_to_mask(
        self,
        image,
        mask,
        trim_image_to_mask=True,
        isolate_mask=True,
        resize_output=True,
        scaling_type="lanczos",
        resolution=512,
        mask_expand=1.0,
        mask_fill_holes=True,
        mask_invert=False,
        context_expand_borders=1.0,
        background="black",
    ):
        device = image.device
        batch, img_h, img_w, _ = image.shape

        # Normalize mask shape to [B, H, W]
        if mask.dim() == 4 and mask.shape[3] == 1:
            mask = mask.squeeze(-1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # Handle batched masks from segmentation nodes (e.g., SAM3)
        # If we have multiple masks (N > 1) but only one image, combine masks
        mask_batch = mask.shape[0]
        if mask_batch > 1 and batch == 1:
            print(f"[SCG Trim] Combining {mask_batch} masks into single composite mask")
            mask = self._combine_masks(mask)
        
        if mask.shape[0] != batch:
            # Broadcast single mask across batch if needed
            mask = mask.expand(batch, -1, -1)

        background_value = 0.0 if background in ["black", "transparent"] else 1.0

        # If not trimming, just apply mask to full image and return
        if not trim_image_to_mask:
            processed_images = []
            processed_masks = []
            
            for i in range(batch):
                img_sample = image[i]
                mask_sample = mask[i].float()
                
                # Prepare mask (fill holes, dilate, invert as configured)
                mask_np = mask_sample.detach().cpu().numpy()
                mask_np = (mask_np >= 0.5).astype(np.float32)
                mask_np = self._fill_mask_holes(mask_np, mask_fill_holes)
                mask_processed = torch.from_numpy(mask_np).to(device=device, dtype=img_sample.dtype)
                
                # Dilate mask if requested
                if mask_expand > 1.0:
                    bbox = self._compute_bbox(mask_processed)
                    if bbox is not None:
                        x_min, y_min, x_max, y_max = bbox
                        base_w = x_max - x_min + 1
                        base_h = y_max - y_min + 1
                        mask_processed = self._dilate_mask(mask_processed, mask_expand, base_w, base_h)
                
                # Invert mask if requested
                if mask_invert:
                    mask_processed = 1.0 - mask_processed
                
                # Apply mask to image if isolate_mask is True
                if isolate_mask:
                    mask_3c = mask_processed.unsqueeze(-1)
                    result_img = img_sample * mask_3c + background_value * (1.0 - mask_3c)
                else:
                    result_img = img_sample.clone()
                
                processed_images.append(result_img)
                processed_masks.append(mask_processed)
            
            images_batch = torch.stack(processed_images, dim=0)
            masks_batch = torch.stack(processed_masks, dim=0)
            
            # Create minimal restitch_data (no cropping was done)
            restitch_data = {
                "orig_image": image.clone(),
                "bbox_x": torch.zeros(batch, device=device),
                "bbox_y": torch.zeros(batch, device=device),
                "crop_w": torch.full((batch,), img_w, device=device),
                "crop_h": torch.full((batch,), img_h, device=device),
                "pad_top": torch.zeros(batch, device=device),
                "pad_left": torch.zeros(batch, device=device),
                "padded_h": torch.full((batch,), img_h, device=device),
                "padded_w": torch.full((batch,), img_w, device=device),
                "final_h": img_h,
                "final_w": img_w,
                "resize_applied": False,
                "scaling_type": scaling_type,
                "mask": masks_batch,
            }
            
            return images_batch, masks_batch, restitch_data

        processed = []
        max_h = 0
        max_w = 0

        for i in range(batch):
            img_sample = image[i]
            mask_sample = mask[i].float()

            # Prepare binary mask for bbox (fill holes, no inversion yet)
            mask_np = mask_sample.detach().cpu().numpy()
            mask_np = (mask_np >= 0.5).astype(np.float32)
            mask_np = self._fill_mask_holes(mask_np, mask_fill_holes)
            mask_for_bbox = torch.from_numpy(mask_np).to(device=device, dtype=img_sample.dtype)

            bbox = self._compute_bbox(mask_for_bbox)
            if bbox is None:
                # No mask content: emit a 1x1 black tile and empty mask
                empty_img = torch.full((1, 1, 3), background_value, device=device, dtype=img_sample.dtype)
                empty_mask = torch.zeros((1, 1), device=device, dtype=img_sample.dtype)
                processed.append(
                    {
                        "trimmed": empty_img,
                        "mask_crop": empty_mask,
                        "bbox_x": 0,
                        "bbox_y": 0,
                        "crop_w": 1,
                        "crop_h": 1,
                    }
                )
                max_h = max(max_h, 1)
                max_w = max(max_w, 1)
                continue

            x_min, y_min, x_max, y_max = bbox
            base_w = x_max - x_min + 1
            base_h = y_max - y_min + 1

            # Dilate mask to give edges breathing room
            dilated_mask = self._dilate_mask(mask_for_bbox, mask_expand, base_w, base_h)
            bbox = self._compute_bbox(dilated_mask)
            if bbox is None:
                bbox = (x_min, y_min, x_max, y_max)

            # Expand context borders after dilation
            x_min, y_min, x_max, y_max, box_w, box_h = self._expand_bbox(
                bbox, context_expand_borders, img_w, img_h
            )

            # Crop image and mask
            img_crop = img_sample[y_min : y_max + 1, x_min : x_max + 1, :].clone()
            mask_crop = dilated_mask[y_min : y_max + 1, x_min : x_max + 1].clone()

            if mask_invert:
                mask_crop = 1.0 - mask_crop

            # Apply mask to create background
            if isolate_mask:
                mask_3c = mask_crop.unsqueeze(-1)
                trimmed = img_crop * mask_3c + background_value * (1.0 - mask_3c)
            else:
                # Only trim by bbox; keep original content inside the crop
                trimmed = img_crop.clone()

            processed.append(
                {
                    "trimmed": trimmed,
                    "mask_crop": mask_crop,
                    "bbox_x": x_min,
                    "bbox_y": y_min,
                    "crop_w": trimmed.shape[1],
                    "crop_h": trimmed.shape[0],
                }
            )
            max_h = max(max_h, trimmed.shape[0])
            max_w = max(max_w, trimmed.shape[1])

        # Pad to consistent batch shape
        padded_images = []
        padded_masks = []
        for item in processed:
            img_crop = item["trimmed"]
            mask_crop = item["mask_crop"]
            h, w = img_crop.shape[0], img_crop.shape[1]
            pad_top = (max_h - h) // 2
            pad_bottom = max_h - h - pad_top
            pad_left = (max_w - w) // 2
            pad_right = max_w - w - pad_left

            img_padded = torch.full(
                (max_h, max_w, 3),
                background_value,
                device=device,
                dtype=img_crop.dtype,
            )
            img_padded[pad_top : pad_top + h, pad_left : pad_left + w, :] = img_crop

            mask_padded = torch.zeros(
                (max_h, max_w),
                device=device,
                dtype=mask_crop.dtype,
            )
            mask_padded[pad_top : pad_top + h, pad_left : pad_left + w] = mask_crop

            padded_images.append(img_padded)
            padded_masks.append(mask_padded)

            item["pad_top"] = pad_top
            item["pad_left"] = pad_left
            item["padded_h"] = max_h
            item["padded_w"] = max_w

        images_batch = torch.stack(padded_images, dim=0)
        masks_batch = torch.stack(padded_masks, dim=0)

        final_h, final_w = images_batch.shape[1], images_batch.shape[2]
        resize_applied = False

        if resize_output:
            current_h, current_w = images_batch.shape[1], images_batch.shape[2]
            if current_h != 0 and current_w != 0:
                scale = resolution / max(current_h, current_w)
                target_h = max(1, int(round(current_h * scale)))
                target_w = max(1, int(round(current_w * scale)))

                images_batch = self._resize_images(images_batch, target_h, target_w, scaling_type)
                masks_batch = self._resize_masks(masks_batch, target_h, target_w)
                resize_applied = True
                final_h, final_w = target_h, target_w

        # Build restitch payload per-sample (share arrays for efficiency)
        bbox_x = []
        bbox_y = []
        crop_w = []
        crop_h = []
        pad_t = []
        pad_l = []
        padded_h = []
        padded_w = []
        orig_images = []
        for idx, item in enumerate(processed):
            bbox_x.append(item["bbox_x"])
            bbox_y.append(item["bbox_y"])
            crop_w.append(item["crop_w"])
            crop_h.append(item["crop_h"])
            pad_t.append(item["pad_top"])
            pad_l.append(item["pad_left"])
            padded_h.append(item["padded_h"])
            padded_w.append(item["padded_w"])
            orig_images.append(image[idx])

        restitch_data = {
            "orig_image": torch.stack(orig_images, dim=0),
            "bbox_x": torch.tensor(bbox_x, device=device),
            "bbox_y": torch.tensor(bbox_y, device=device),
            "crop_w": torch.tensor(crop_w, device=device),
            "crop_h": torch.tensor(crop_h, device=device),
            "pad_top": torch.tensor(pad_t, device=device),
            "pad_left": torch.tensor(pad_l, device=device),
            "padded_h": torch.tensor(padded_h, device=device),
            "padded_w": torch.tensor(padded_w, device=device),
            "final_h": final_h,
            "final_w": final_w,
            "resize_applied": resize_applied,
            "scaling_type": scaling_type,
            "mask": masks_batch,
        }

        return images_batch, masks_batch, restitch_data


class SCGFlipBoolean:
    """
    A simple utility node that flips a boolean value.
    Accepts a boolean input and outputs the inverted boolean.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "flip_boolean"
    CATEGORY = "scg-utils"
    
    def flip_boolean(self, boolean):
        """
        Flip the input boolean value.
        
        Args:
            boolean: Input boolean value
        
        Returns:
            Tuple containing the flipped boolean
        """
        return (not boolean,)


class SCGFormatInteger:
    """
    A utility node that rounds an integer to a specified divisible value
    and optionally adds a constant after rounding.
    
    Useful for ensuring dimensions are divisible by common values (8, 16, 32, etc.)
    for model requirements.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 2147483647}),
                "add_to_final": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647}),
                "round_mode": (["round_up", "round_down"], {"default": "round_down"}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "format_integer"
    CATEGORY = "scg-utils"
    
    def format_integer(self, value, divisible_by, add_to_final, round_mode):
        """
        Round an integer to the nearest multiple of divisible_by, then add add_to_final.
        
        Args:
            value: Input integer to format
            divisible_by: The number the result should be divisible by
            add_to_final: Value to add after rounding (applied after rounding)
            round_mode: Whether to round up or down to the nearest divisible
        
        Returns:
            Tuple containing the formatted integer
        """
        if round_mode == "round_up":
            # Round up to next multiple of divisible_by
            rounded = math.ceil(value / divisible_by) * divisible_by
        else:
            # Round down to previous multiple of divisible_by
            rounded = math.floor(value / divisible_by) * divisible_by
        
        # Add the final value after rounding
        result = rounded + add_to_final
        
        return (result,)


class SCGStitchInpaintImage:
    """
    Stitch an inpainted crop (from SCG Trim Image to Mask) back into the original
    image using the stored restitch metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "restitch_data": ("RESTITCH",),
                "image": ("IMAGE",),
                "stitch_mask": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "scg-utils"

    def _resize_images(self, images, target_h, target_w, method):
        if method == "lanczos":
            resized = []
            for img in images:
                pil_img = Image.fromarray(
                    (img.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                )
                pil_resized = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)
                resized.append(
                    torch.from_numpy(np.array(pil_resized).astype(np.float32) / 255.0).to(img.device)
                )
            return torch.stack(resized, dim=0)

        bchw = images.permute(0, 3, 1, 2)
        if method in ["bilinear", "bicubic"]:
            resized_bchw = torch.nn.functional.interpolate(
                bchw, size=(target_h, target_w), mode=method, align_corners=False, antialias=True
            )
        elif method == "area":
            resized_bchw = torch.nn.functional.interpolate(
                bchw, size=(target_h, target_w), mode=method
            )
        else:
            resized_bchw = torch.nn.functional.interpolate(
                bchw, size=(target_h, target_w), mode=method
            )
        return resized_bchw.permute(0, 2, 3, 1)

    def _resize_masks(self, masks, target_h, target_w):
        masks = masks.unsqueeze(1)
        resized = torch.nn.functional.interpolate(masks, size=(target_h, target_w), mode="nearest")
        return resized.squeeze(1)

    def stitch(self, restitch_data, image, stitch_mask=True):
        orig_images = restitch_data["orig_image"]
        bbox_x = restitch_data["bbox_x"]
        bbox_y = restitch_data["bbox_y"]
        crop_w = restitch_data["crop_w"]
        crop_h = restitch_data["crop_h"]
        pad_top = restitch_data["pad_top"]
        pad_left = restitch_data["pad_left"]
        padded_h = restitch_data["padded_h"]
        padded_w = restitch_data["padded_w"]
        final_mask = restitch_data["mask"]

        scaling_type = restitch_data.get("scaling_type", "lanczos")
        resize_applied = restitch_data.get("resize_applied", False)
        final_h = restitch_data.get("final_h", image.shape[1])
        final_w = restitch_data.get("final_w", image.shape[2])

        batch = image.shape[0]
        results = []

        for idx in range(batch):
            ref_idx = idx if idx < orig_images.shape[0] else orig_images.shape[0] - 1
            base_img = orig_images[ref_idx].clone()
            patch = image[idx]

            # Ensure input matches expected final size
            if patch.shape[0] != final_h or patch.shape[1] != final_w:
                patch = self._resize_images(patch.unsqueeze(0), final_h, final_w, scaling_type)[0]

            # Only process mask if stitch_mask is True
            if stitch_mask:
                mask_resized = final_mask[ref_idx].clone()
                if mask_resized.shape[0] != final_h or mask_resized.shape[1] != final_w:
                    mask_resized = self._resize_masks(mask_resized.unsqueeze(0), final_h, final_w)[0]

            # Undo resize to padded size if needed
            if resize_applied:
                target_h = int(padded_h[ref_idx].item())
                target_w = int(padded_w[ref_idx].item())
                patch = self._resize_images(patch.unsqueeze(0), target_h, target_w, scaling_type)[0]
                if stitch_mask:
                    mask_resized = self._resize_masks(mask_resized.unsqueeze(0), target_h, target_w)[0]
            else:
                target_h = final_h
                target_w = final_w

            # Remove padding to original crop size
            pt = int(pad_top[ref_idx].item())
            pl = int(pad_left[ref_idx].item())
            cw = int(crop_w[ref_idx].item())
            ch = int(crop_h[ref_idx].item())

            patch = patch[pt : pt + ch, pl : pl + cw, :]

            x = int(bbox_x[ref_idx].item())
            y = int(bbox_y[ref_idx].item())

            # Blend into original
            if stitch_mask:
                mask_crop = mask_resized[pt : pt + ch, pl : pl + cw]
                patch_mask = mask_crop.unsqueeze(-1)
                target_slice = base_img[y : y + ch, x : x + cw, :]
                blended = target_slice * (1.0 - patch_mask) + patch * patch_mask
                base_img[y : y + ch, x : x + cw, :] = blended
            else:
                # Just paste the entire patch without mask blending
                base_img[y : y + ch, x : x + cw, :] = patch

            results.append(base_img)

        return (torch.stack(results, dim=0),)