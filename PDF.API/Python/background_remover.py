import os
import sys
import glob
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

# U2NET architecture components
class REBNCONV(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = torch.nn.BatchNorm2d(out_ch)
        self.relu_s1 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))

# U2NET RSU-7 block
class RSU7(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=True)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

# U2NET RSU-6 block
class RSU6(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

# U2NET RSU-5 block
class RSU5(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

# U2NET RSU-4 block
class RSU4(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin
    
# U2NET RSU-4F block
class RSU4F(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin

# U2NET model
class U2NET(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 256, 512)
        
        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        # Side outputs
        self.side1 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = torch.nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = torch.nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = torch.nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = torch.nn.Conv2d(512, out_ch, 3, padding=1)
        
        self.outconv = torch.nn.Conv2d(6*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        
        # Stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        # Stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        # Stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        # Stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        # Stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        # Stage 6
        hx6 = self.stage6(hx)
        
        # Decoder
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=True)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # Side outputs
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=True)
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=True)
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

# Model loading and processing functions
def find_local_model(model_name='u2net'):
    """Look for model in Models folder"""
    # Check if model exists in Models folder
    model_path = os.path.join('Saved_Models', f"{model_name}.pth")
    if os.path.exists(model_path):
        print(f"Found local model at {model_path}")
        return model_path
    return None

def load_model(model_dir='Saved_Models', model_name='u2net'):
    try:
        # First check if the model exists in the Models folder
        local_model_path = find_local_model(model_name)
        
        if local_model_path:
            # Load the local model from Models folder
            print(f"Loading model from {local_model_path}")
            net = U2NET()
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(local_model_path))
                net.cuda()
            else:
                net.load_state_dict(torch.load(local_model_path, map_location='cpu'))
            net.eval()
            return net
        
        # If not in Models folder, check saved_models directory
        model_path = os.path.join(model_dir, model_name, f"{model_name}.pth")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            net = U2NET()
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(model_path))
                net.cuda()
            else:
                net.load_state_dict(torch.load(model_path, map_location='cpu'))
            net.eval()
            return net
        
        raise Exception(f"Model {model_name} not found in Models folder or {model_dir} directory")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise Exception(f"Failed to load model: {model_name}")


def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((320, 320), Image.BILINEAR)
    image = np.array(image) / 255.0
    
    # If grayscale, convert to RGB
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    
    # Handle RGBA images
    elif image.shape[2] == 4:
        # Use alpha channel if available
        alpha = image[:, :, 3:4]
        image = image[:, :, 0:3] * alpha + (1 - alpha)
    
    # Transpose to channel-first format and add batch dimension
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    
    return image_tensor

def predict_mask(model, image_tensor):
    # PyTorch model
    # Move to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = outputs[0]
    
    # Convert prediction to numpy array
    if torch.cuda.is_available():
        pred = pred.cpu()
    pred = pred.numpy()
    
    # Post-process prediction
    pred = pred.squeeze()
    pred = (pred * 255).astype(np.uint8)
    
    return pred

def remove_background(image_path, output_path=None, model=None):
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    
    # Get original image size
    original_size = image.size
    
    # Preprocess image
    image_tensor = preprocess_image(image)
    
    # Predict mask
    mask = predict_mask(model, image_tensor)
    
    # Resize mask to original image size
    mask = Image.fromarray(mask).resize(original_size, Image.BILINEAR)
    mask = np.array(mask)
    
    # Create alpha mask
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    
    # Load original image
    original_image = np.array(Image.open(image_path).convert('RGBA'))
    
    # Create RGBA image with alpha from mask
    if original_image.shape[2] == 4:
        original_image[:, :, 3] = mask
    else:
        original_image = np.concatenate([original_image, mask[:, :, np.newaxis]], axis=2)
    
    # Create output image with white background
    white_bg = np.ones_like(original_image) * 255
    alpha = mask_3d
    output_image = original_image[:, :, :3] * alpha + white_bg[:, :, :3] * (1 - alpha)
    output_image = output_image.astype(np.uint8)
    
    result_image = Image.fromarray(output_image)
    
    # Save result if output path is provided
    if output_path:
        result_image.save(output_path)
        print(f"Saved result to {output_path}")
    
    return result_image

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Remove background from images')
    parser.add_argument('--input', type=str, required=True, help='Input image path or directory')
    parser.add_argument('--output', type=str, help='Output image path or directory (optional)')
    parser.add_argument('--model', type=str, default='u2net', choices=['u2net', 'u2netp'], help='Model to use (u2net or u2netp)')
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Load model
        model = load_model(model_name=args.model)
        
        # Process single file
        print(f"Processing {args.input}...")
        
        # Generate output path if not provided
        if not args.output:
            output_filename = os.path.splitext(os.path.basename(args.input))[0] + "_nobg.png"
            output_path = os.path.join(os.path.dirname(args.input), output_filename)
        else:
            output_path = args.output
        
        try:
            # Process image
            remove_background(args.input, output_path, model)
            print(f"Successfully processed: {args.input}")
        except Exception as e:
            print(f"Error processing {args.input}: {e}")
    
    elif os.path.isdir(args.input):
        # Process directory
        print(f"Processing directory: {args.input}")
        
        # Create output directory if not provided
        if not args.output:
            output_dir = os.path.join(args.input, "no_bg")
        else:
            output_dir = args.output
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model once for all images
        model = load_model(model_name=args.model)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(args.input, f"*{ext.upper()}")))
        
        if not image_files:
            print(f"No image files found in {args.input}")
            sys.exit(1)
        
        # Process each image with progress reporting
        total_files = len(image_files)
        successful = 0
        failed = 0
        
        print(f"Found {total_files} images to process")
        
        for i, img_path in enumerate(image_files):
            # Generate output filename
            filename = os.path.splitext(os.path.basename(img_path))[0] + "_nobg.png"
            output_path = os.path.join(output_dir, filename)
            
            # Progress update
            progress = (i + 1) / total_files * 100
            print(f"[{i+1}/{total_files}] ({progress:.1f}%) Processing: {img_path}")
            
            try:
                remove_background(img_path, output_path, model)
                successful += 1
            except Exception as e:
                print(f"  Error: {e}")
                failed += 1
        
        # Report final statistics
        print("\nProcessing complete!")
        print(f"Total files: {total_files}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        sys.exit(1)
