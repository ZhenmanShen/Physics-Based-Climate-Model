# light_unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_x1, in_channels_x2, out_channels, bilinear=True):
        """
        Args:
            in_channels_x1 (int): Number of channels in the feature map from the deeper layer (to be upsampled).
            in_channels_x2 (int): Number of channels in the feature map from the skip connection.
            out_channels (int): Desired number of output channels for this Up block.
            bilinear (bool): Whether to use bilinear upsampling or transposed convolutions.
        """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concatenation, input channels to DoubleConv will be in_channels_x1 + in_channels_x2
            # We can set mid_channels for DoubleConv, e.g., to out_channels or (sum_in_channels)//2
            self.conv = DoubleConv(in_channels_x1 + in_channels_x2, out_channels, mid_channels=out_channels) # Or (in_channels_x1 + in_channels_x2) // 2
        else:
            # ConvTranspose2d typically halves the channels if out_channels is not specified, or maps to a specified number.
            # Here, we want ConvTranspose2d to output in_channels_x1 // 2 (or a pre-calculated value)
            # so that after concat with in_channels_x2, it matches an expected sum for DoubleConv.
            # A common strategy is for ConvTranspose2d to output channels equal to in_channels_x2.
            # Or, more simply, ConvTranspose2d(in_channels_x1, target_channels_after_transpose_conv, ...)
            # Let's assume ConvTranspose2d outputs in_channels_x1 // 2 channels to mirror typical U-Net structure
            self.up = nn.ConvTranspose2d(in_channels_x1, in_channels_x1 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels_x1 // 2) + in_channels_x2, out_channels)

    def forward(self, x1, x2):
        # x1 is from the upsampling path (deeper layer), x2 is the skip connection from the encoder
        x1 = self.up(x1)
        
        # Pad x1 if its spatial dimensions are smaller than x2's after upsampling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY > 0 or diffX > 0: # Only pad if necessary
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LightUNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_channels=32, depth=3, bilinear=True):
        super(LightUNet, self).__init__()
        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_input_channels, base_channels)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        encoder_channels = [base_channels] # Channels after each DoubleConv in encoder
        current_enc_channels = base_channels
        for _ in range(depth):
            self.down_blocks.append(Down(current_enc_channels, current_enc_channels * 2))
            current_enc_channels *= 2
            encoder_channels.append(current_enc_channels)
        # encoder_channels will be [base, base*2, base*4, ..., base*(2**depth)]
        # The last element is the bottleneck channels.
        # Skip connections should come from before maxpool, so they are encoder_channels[:-1]

        # Decoder
        self.up_blocks = nn.ModuleList()
        # ch_from_deeper_layer starts as bottleneck channels
        ch_from_deeper_layer = encoder_channels[depth] # This is base_channels * (2**depth)
        
        for i in range(depth):
            # Skip connection channels are from the corresponding encoder stage
            # encoder_channels are [C0, C1, C2, C3] for depth=3. Skips are C0, C1, C2.
            # For up_block i=0 (deepest), skip is encoder_channels[depth-1-i] = encoder_channels[2]
            # For up_block i=1, skip is encoder_channels[depth-1-i] = encoder_channels[1]
            # For up_block i=2, skip is encoder_channels[depth-1-i] = encoder_channels[0]
            ch_from_skip_connection = encoder_channels[depth - 1 - i]
            
            # Output channels of this Up block
            # This should typically be ch_from_skip_connection to match U-Net structure
            out_ch_for_this_up_stage = ch_from_skip_connection 
            
            self.up_blocks.append(
                Up(in_channels_x1=ch_from_deeper_layer, 
                   in_channels_x2=ch_from_skip_connection, 
                   out_channels=out_ch_for_this_up_stage, 
                   bilinear=bilinear)
            )
            ch_from_deeper_layer = out_ch_for_this_up_stage # Update for next iteration
            
        self.outc = OutConv(base_channels, n_output_channels) # Input to OutConv is output of last Up block

    def forward(self, x):
        # Encoder
        skip_outputs = []
        current_x = self.inc(x) # Initial convolution
        skip_outputs.append(current_x) # Store output of initial DoubleConv

        for i, down_block in enumerate(self.down_blocks):
            # MaxPool is inside down_block, so current_x becomes input to DoubleConv after MaxPool
            # We need to store the output of DoubleConv *before* MaxPool for skip connections
            # The current `Down` block does MaxPool then DoubleConv.
            # So, the output of `inc` is the first skip.
            # The output of each `DoubleConv` in `Down` blocks are the subsequent skips.
            # This means `skip_outputs` should store `current_x` *before* it goes into `down_block`'s MaxPool,
            # but *after* the `DoubleConv` of the previous stage.

            # Let's adjust the encoder logic slightly for clarity on skips
            # x_stages[0] = self.inc(x)
            # x_stages[1] = self.down_blocks[0].double_conv(self.down_blocks[0].maxpool(x_stages[0]))
            # This is getting complicated. Let's simplify how skips are collected.

            # Simpler skip collection:
            # current_x is the output of the DoubleConv at each stage *before* pooling for the next.
            # So, the stored skip_outputs are correct.
            current_x = down_block(current_x) # current_x is now output of DoubleConv in Down block
            if i < len(self.down_blocks) -1: # For all but the one leading to bottleneck
                 skip_outputs.append(current_x)
        # current_x is now the bottleneck

        # Decoder
        # skip_outputs are [inc_out, down0_out, down1_out, ...] (excluding bottleneck)
        # We need them in reverse order for upsampling.
        for i, up_block in enumerate(self.up_blocks):
            skip_connection_tensor = skip_outputs[len(skip_outputs) - 1 - i]
            current_x = up_block(current_x, skip_connection_tensor)
            
        logits = self.outc(current_x)
        return logits

if __name__ == '__main__':
    n_inputs = 5
    n_outputs = 2
    img_height = 48
    img_width = 72
    batch_size = 4

    configs_to_test = [
        {"name": "D3_B32 (Original Problematic)", "base_channels": 32, "depth": 3, "bilinear": True}, # ~1.8M
        {"name": "D2_B32 (Lighter)", "base_channels": 32, "depth": 2, "bilinear": True},      # Target: < 700k
        {"name": "D2_B24 (Even Lighter)", "base_channels": 24, "depth": 2, "bilinear": True}, # Target: < 500k
        {"name": "D2_B16 (Very Light)", "base_channels": 16, "depth": 2, "bilinear": True},   # Target: < 300k
        {"name": "D1_B32 (Super Light)", "base_channels": 32, "depth": 1, "bilinear": True},
    ]

    for config in configs_to_test:
        print(f"\n--- Testing Config: {config['name']} ---")
        model = LightUNet(
            n_input_channels=n_inputs, 
            n_output_channels=n_outputs, 
            base_channels=config['base_channels'], 
            depth=config['depth'], 
            bilinear=config['bilinear']
        )
        dummy_input = torch.randn(batch_size, n_inputs, img_height, img_width)
        
        try:
            output = model(dummy_input)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")
            print(f"Number of parameters: {num_params:,}")
            if num_params > 700000 and "Original" not in config["name"]:
                 print("WARNING: Still potentially too large.")
            elif "Original" not in config["name"]:
                 print("Parameter count looks good for 'lighter'.")

        except Exception as e:
            print(f"Error with config {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Expected output for D2_B16 should be around 200-250k params.
    # D2_B32:
    # inc: DC(5,32)
    # d1: Down(32,64) -> DC(32,64)
    # d2_bottleneck: Down(64,128) -> DC(64,128)
    # up1: Up(128, 64, 64) -> DC(128+64=192, 64)
    # up2: Up(64, 32, 32) -> DC(64+32=96, 32)
    # outc: OC(32,2)
