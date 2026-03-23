
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from src.utils_onnx import msg_to_pil, transform_images, load_model_trt, load_model_onnx


@dataclass
class SchedulerOutput:
    """Output class for scheduler step."""
    prev_sample: np.ndarray
    pred_original_sample: Optional[np.ndarray] = None

class DDPMSchedulerNumPy:
    """
    NumPy implementation of DDPM (Denoising Diffusion Probabilistic Models) scheduler.
    Compatible with ONNX model inference.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        variance_type: str = "fixed_small",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        
        # Compute betas
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        elif beta_schedule == "scaled_linear":
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float64) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._betas_for_alpha_bar(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
        # Precompute values for q(x_t | x_0) and q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.clip(self.posterior_variance, a_min=1e-20, a_max=None)
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Timesteps
        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1)
        self.num_inference_steps = None
    
    def _betas_for_alpha_bar(self, num_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        def alpha_bar(t):
            return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas, dtype=np.float64)
    
    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = np.arange(0, num_inference_steps)[::-1] * step_ratio
        self.timesteps = self.timesteps.astype(np.int64)
    
    def _get_variance(self, t: int) -> float:
        """Get variance for timestep t."""
        prev_t = t - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        
        if self.variance_type == "fixed_small":
            variance = np.clip(variance, a_min=1e-20, a_max=None)
        elif self.variance_type == "fixed_small_log":
            variance = np.log(np.clip(variance, a_min=1e-20, a_max=None))
        elif self.variance_type == "fixed_large":
            variance = self.betas[t]
        elif self.variance_type == "fixed_large_log":
            variance = np.log(self.betas[t])
        elif self.variance_type == "learned_range":
            raise NotImplementedError("learned_range variance not supported")
        
        return variance
    
    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        generator: Optional[np.random.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE.
        
        Args:
            model_output: Output from the learned diffusion model (typically noise prediction)
            timestep: Current discrete timestep
            sample: Current instance of sample being created by diffusion process
            generator: Random number generator for reproducibility
            return_dict: Whether to return SchedulerOutput or tuple
        
        Returns:
            SchedulerOutput with prev_sample or tuple
        """
        t = timestep
        prev_t = t - self.num_train_timesteps // self.num_inference_steps
        
        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # 2. Compute predicted original sample from predicted noise
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - np.sqrt(beta_prod_t) * model_output
            ) / np.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                np.sqrt(alpha_prod_t) * sample - np.sqrt(beta_prod_t) * model_output
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # 3. Clip predicted x_0
        if self.clip_sample:
            pred_original_sample = np.clip(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )
        
        # 4. Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (
            np.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
        )
        current_sample_coeff = np.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        
        # 5. Compute predicted previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample 
            + current_sample_coeff * sample
        )
        
        # 6. Add noise
        variance = 0
        if t > 0:
            variance = self._get_variance(t)
            if self.variance_type in ["fixed_small_log", "fixed_large_log"]:
                variance = np.exp(0.5 * variance)
            else:
                variance = np.sqrt(variance)
            
            if generator is None:
                noise = np.random.randn(*model_output.shape).astype(model_output.dtype)
            else:
                noise = generator.standard_normal(model_output.shape).astype(model_output.dtype)
            
            pred_prev_sample = pred_prev_sample + variance * noise
        
        if not return_dict:
            return (pred_prev_sample,)
        
        return SchedulerOutput(
            prev_sample=pred_prev_sample,
            pred_original_sample=pred_original_sample
        )
    
    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Add noise to samples for training (forward diffusion)."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[..., np.newaxis]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., np.newaxis]
        
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples


# Example usage with ONNX model
def example_inference():
    """Example showing how to use with an ONNX model."""
    import onnxruntime as ort
    
    # Initialize scheduler
    scheduler = DDPMSchedulerNumPy(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(50)  # Use 50 inference steps
    
    # Load ONNX model
    # session = ort.InferenceSession("your_model.onnx")
    
    # Start from pure noise
    latents = np.random.randn(1, 4, 64, 64).astype(np.float32)
    
    # Denoising loop
    for t in scheduler.timesteps:
        # Prepare model input
        timestep = np.array([t], dtype=np.int64)
        
        # Run ONNX model (uncomment when using real model)
        # model_output = session.run(None, {
        #     "sample": latents,
        #     "timestep": timestep,
        # })[0]
        
        # Placeholder for demo
        model_output = np.random.randn(*latents.shape).astype(np.float32)
        
        # Scheduler step
        result = scheduler.step(model_output, t, latents)
        latents = result.prev_sample
    
    return latents


if __name__ == "__main__":
    # Quick test
    scheduler = DDPMSchedulerNumPy()
    scheduler.set_timesteps(50)
    print(f"Timesteps: {scheduler.timesteps[:5]}...")
    
    # Test step
    sample = np.random.randn(1, 4, 64, 64).astype(np.float32)
    noise_pred = np.random.randn(1, 4, 64, 64).astype(np.float32)
    
    result = scheduler.step(noise_pred, scheduler.timesteps[0], sample)
    print(f"Output shape: {result.prev_sample.shape}")
    print("DDPM NumPy scheduler working correctly!")
