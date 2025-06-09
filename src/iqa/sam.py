import torch

class SAM:

    @staticmethod
    def evaluate(
        img1: torch.Tensor,
        img2: torch.Tensor,
        eps: float = 1e-8,
        return_degrees: bool = True,
    ) -> torch.Tensor:
        """
        Computes the Spectral Angle Mapper (SAM) between two image tensors.

        :param img1: original image (as torch tensor)
        :param img2: manipulated image (as torch tensor)
        :param eps: small constant to avoid division by zero
        :param return_degrees: whether to return the angle in degrees (default) or radians
        :return: SAM values for each image pair, lower is better
        """

        assert img1.shape == img2.shape, "Input tensors must have the same shape"

        if img1.dim() == 3 and img2.dim() == 3:
            img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)

        N, C, H, W = img1.shape
        img1_flat = img1.view(N, C, -1)
        img2_flat = img2.view(N, C, -1)

        cos_theta = torch.cosine_similarity(img1_flat, img2_flat, dim=1, eps=eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        angle = torch.acos(cos_theta)
        angle_mean = angle.mean(dim=1)

        if return_degrees:
            angle_mean = torch.rad2deg(angle_mean)

        return angle_mean


__all__ = ['SAM']
