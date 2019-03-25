"""
    Loss function for unsupervised learning for stereo image pairs. Include three different loss functions:
    - Appearance matching loss
    - Disparity smoothness loss
    - Left-Right Disparity Consistency Loss
"""
class MonoLoss(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        self.build_losses()
        
