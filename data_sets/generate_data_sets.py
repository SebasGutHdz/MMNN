'''
Implementation of test functions from the papers:
[1] STRUCTURED AND BALANCED MULTI-COMPONENT AND MULTI-LAYER NEURAL NETWORKS
[2] Fourier Multi-Component and Multi-Layer Neural Networks:  Unlocking High-Frequency Potential


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
import math

class TestFunctions:
    """All test functions from the FMMNN paper for numerical experiments"""

    @staticmethod
    def f1_MMNN(x: torch.Tensor) -> torch.Tensor:
        """test function f1 on [1] """
        return torch.cos(20 * torch.pi * torch.abs(x) ** 1.4) + 0.5 * torch.cos(12 * torch.pi * torch.abs(x) ** 1.6)
    
    @staticmethod
    def f2_MMNN(x: torch.Tensor, s: float = 2) -> torch.Tensor:
        """test function f2 on [1] 
        f(x_1,x_2) = sum_{i,j = 1}^2 a_ij * sin(s*b_i*x_i + s*c_{i,j}*x_i*x_j) * cos(s*b_j*x_j + s*d_{i,j}*x_i^2)
        a = [[0.3,0.2],[0.2,0.3]], b=[2π,4π], c=[[2π,4π],[8π,4π]], d=[[4π,6π],[8π,6π]]
        """
        # Coefficient matrices from the paper
        a = torch.tensor([[0.3, 0.2], [0.2, 0.3]])
        b = torch.tensor([2 * torch.pi, 4 * torch.pi])
        c = torch.tensor([[2 * torch.pi, 4 * torch.pi], [8 * torch.pi, 4 * torch.pi]])
        d = torch.tensor([[4 * torch.pi, 6 * torch.pi], [8 * torch.pi, 6 * torch.pi]])

        x1, x2 = x[:, 0], x[:, 1]

        results = torch.zeros_like(x1)

        def apply_fn_to(x: torch.Tensor) -> torch.Tensor:
            """
            Apply the function to a single input point, we then use vmap to vectorize over batch
            x: (2,) tensor
            """
            results = 0.0
            for i in range(2):
                for j in range(2):
                    sin_term = torch.sin(s * b[i] * (x[0] if i == 0 else x[1]) + 
                                       s * c[i,j] * x[0] * x[1])
                    cos_term = torch.cos(s * b[j] * (x[1] if j == 1 else x[0]) + 
                                   s * d[i,j] * (x1**2 if i == 0 else x2**2))
                results += a[i,j] * sin_term * cos_term
            return results
        
        results = torch.vmap(apply_fn_to)(x)

        return results

    @staticmethod
    def smooth_basis_g(x: torch.Tensor) -> torch.Tensor:
        """Smooth basis function g used in f1 definition"""
        def g0(x):
            # g0(x) = exp(-1/x^2) if x > 0, else 0
            mask = x > 0
            result = torch.zeros_like(x)
            result[mask] = torch.exp(-1.0 / (x[mask] ** 2))
            return result
        
        # g(x) = g0(x+1) * g0(1-x) / g0(1)^2
        g0_1 = g0(torch.tensor(1.0))
        return (g0(x + 1) * g0(1 - x)) / (g0_1 ** 2)
    
    @staticmethod
    def f1_smooth_oscillatory(x: torch.Tensor) -> torch.Tensor:
        """
        f1: Complex smooth oscillatory function (C∞)
        f1(x) = 1/(1+2x²) * sum_{i=-n}^n ((-1)^(i mod 3) * (|i|+n)/n * g((2n+1)x - i/(n+1)))
        where n=36, g is the smooth basis function
        """
        n = 36
        x = x.flatten()
        result = torch.zeros_like(x)
        
        for i in range(-n, n+1):
            coeff = ((-1) ** (abs(i) % 3)) * (abs(i) + n) / n
            arg = (2*n + 1) * x - i / (n + 1)
            g_val = TestFunctions.smooth_basis_g(arg)
            result += coeff * g_val
        
        return result / (1 + 2 * x**2)
    
    @staticmethod
    def f2_nonsmooth_oscillatory(x: torch.Tensor) -> torch.Tensor:
        """
        f2: Non-smooth oscillatory function 
        f2(x) = (1+6x⁸)/(1+8x⁶) * (⌊120x²-2⌋ + ⌊120x²+1⌋/2)²
        """
        x = x.flatten()
        numerator = 1 + 6 * x**8
        denominator = 1 + 8 * x**6
        
        term1 = torch.floor((120*x**2+1)/2)
        term2 = (120*x**2-2*term1)**2

        return (numerator / denominator)*term2

    @staticmethod
    def f3_nonsmooth_oscillatory(x: torch.Tensor) -> torch.Tensor:
        """
        f3: Another non-smooth oscillatory function 
        f3(x) = (1+6x⁸)/(1+8x⁶) * (⌊32x-2⌋ + ⌊32x+1⌋/2)²
        """
        x = x.flatten()
        numerator = 1 + 6 * x**8
        denominator = 1 + 8 * x**6
        
        term1 = torch.floor(32 * x - 2)
        term2 = torch.floor(32 * x + 1) / 2
        
        return (numerator / denominator) * (term1 + term2)**2
    
    @staticmethod
    def f_2d_oscillatory_v1(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        2D oscillatory function (simple version)
        f(x1,x2) = cos(20π|x1|^1.4) + 0.5*cos(12π|x2|^1.6)
        """
        term1 = torch.cos(20 * np.pi * torch.abs(x1)**1.4)
        term2 = 0.5 * torch.cos(12 * np.pi * torch.abs(x2)**1.6)
        return term1 + term2
    
    @staticmethod
    def f_2d_oscillatory_v2(x1: torch.Tensor, x2: torch.Tensor, s: float = 2.0) -> torch.Tensor:
        """
        2D oscillatory function (complex version)
        f(x1,x2) = sum_{i,j} a_ij * sin(s*b_i*x_i + s*c_{i,j}*x_i*x_j) * cos(s*b_j*x_j + s*d_{i,j}*x_i²)
        """
        # Coefficient matrices from the paper
        a = torch.tensor([[0.3, 0.2], [0.2, 0.3]])
        b = torch.tensor([2*np.pi, 4*np.pi])
        c = torch.tensor([[2*np.pi, 4*np.pi], [8*np.pi, 4*np.pi]])
        d = torch.tensor([[4*np.pi, 6*np.pi], [8*np.pi, 6*np.pi]])
        
        result = torch.zeros_like(x1)
        for i in range(2):
            for j in range(2):
                sin_term = torch.sin(s * b[i] * (x1 if i == 0 else x2) + 
                                   s * c[i,j] * x1 * x2)
                cos_term = torch.cos(s * b[j] * (x2 if j == 1 else x1) + 
                                   s * d[i,j] * (x1**2 if i == 0 else x2**2))
                result += a[i,j] * sin_term * cos_term
        
        return result
    
    @staticmethod
    def f_3d_oscillatory_v1(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """
        3D oscillatory function
        f(x1,x2,x3) = sum_{i,j} ea_ij * sin(eb_i*x_i + ec_{i,j}*x_i*x_j) * cos(eb_j*x_j + de_{i,j}*x_i²)
        """
        # Extended coefficient matrices for 3D
        ea = torch.tensor([[0.3, 0.1, 0.4], 
                          [0.2, 0.3, 0.1], 
                          [0.2, 0.1, 0.3]])
        eb = torch.tensor([np.pi, 4*np.pi, 3*np.pi])
        ec = torch.tensor([[2*np.pi, np.pi, 3*np.pi],
                          [2*np.pi, 3*np.pi, 2*np.pi],
                          [3*np.pi, np.pi, np.pi]])
        de = torch.tensor([[2*np.pi, 3*np.pi, np.pi],
                          [np.pi, 3*np.pi, 2*np.pi],
                          [np.pi, 2*np.pi, 3*np.pi]])
        
        x = torch.stack([x1, x2, x3], dim=-1)
        result = torch.zeros_like(x1)
        
        for i in range(3):
            for j in range(3):
                sin_term = torch.sin(eb[i] * x[..., i] + ec[i,j] * x[..., i] * x[..., j])
                cos_term = torch.cos(eb[j] * x[..., j] + de[i,j] * x[..., i]**2)
                result += ea[i,j] * sin_term * cos_term
        
        return result
    
    @staticmethod
    def f_3d_polar_levelset(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        3D function in polar coordinates with level sets
        f(r,θ,φ) = piecewise function based on 0.5 + 5ρ - 5r
        where ρ(θ,φ) = 0.5 + 0.2*sin(6θ)*cos(6φ)*sin²(θ)
        """
        rho = 0.5 + 0.2 * torch.sin(6*theta) * torch.cos(6*phi) * torch.sin(theta)**2
        criterion = 0.5 + 5*rho - 5*r
        
        result = torch.zeros_like(r)
        result[criterion <= 0] = 0.0
        result[criterion >= 1] = 1.0
        mask = (criterion > 0) & (criterion < 1)
        result[mask] = criterion[mask]
        
        return result
    
    @staticmethod
    def f_2d_polar_levelset(r: torch.Tensor, theta: torch.Tensor, version: int = 1) -> torch.Tensor:
        """
        2D polar coordinate level set functions
        """
        if version == 1:
            # Version 1: ρ = 0.5 + 0.1*cos(π²θ/2)
            rho = 0.5 + 0.1 * torch.cos(np.pi**2 * theta / 2)
            criterion = 0.5 + 5*rho - 5*r
        else:
            # Version 2: ρ = 0.1 + 0.02*cos(8πθ)  
            rho = 0.1 + 0.02 * torch.cos(8*np.pi*theta)
            criterion = 0.5 + 25*rho - 25*r
        
        result = torch.zeros_like(r)
        result[criterion <= 0] = 0.0
        result[criterion >= 1] = 1.0
        mask = (criterion > 0) & (criterion < 1)
        result[mask] = criterion[mask]
        
        return result
    
    @staticmethod
    def f_4d_gaussian_pdf(x: torch.Tensor, mu: torch.Tensor = None, 
                         sigma_inv: torch.Tensor = None) -> torch.Tensor:
        """
        4D Gaussian probability density function
        f(x) = exp(-0.5*(x-μ)ᵀΣ⁻¹(x-μ)) / sqrt((2π)⁴ * det(Σ))
        """
        if mu is None:
            mu = torch.zeros(4)
        if sigma_inv is None:
            # Use the covariance matrix from the paper
            sigma_inv = 20 * torch.tensor([[1.0, 0.9, 0.8, 0.7],
                                          [0.9, 2.0, 1.9, 1.8], 
                                          [0.8, 1.9, 3.0, 2.9],
                                          [0.7, 1.8, 2.9, 4.0]])
        
        d = x.shape[-1]  # Should be 4
        x_centered = x - mu
        
        # Compute quadratic form
        quad_form = torch.sum(x_centered @ sigma_inv * x_centered, dim=-1)
        
        # Compute determinant for normalization
        det_sigma = 1.0 / torch.det(sigma_inv)
        norm_const = 1.0 / torch.sqrt((2*np.pi)**d * det_sigma)
        
        return norm_const * torch.exp(-0.5 * quad_form)
    
    @staticmethod
    def get_test_function(name: str) -> Callable:
        """Get test function by name for easy access"""
        function_map = {
            'f1': TestFunctions.f1_smooth_oscillatory,
            'f2': TestFunctions.f2_nonsmooth_oscillatory,
            'f3': TestFunctions.f3_nonsmooth_oscillatory,
            'f_2d_v1': TestFunctions.f_2d_oscillatory_v1,
            'f_2d_v2': TestFunctions.f_2d_oscillatory_v2,
            'f_3d_v1': TestFunctions.f_3d_oscillatory_v1,
            'f_3d_polar': TestFunctions.f_3d_polar_levelset,
            'f_2d_polar': TestFunctions.f_2d_polar_levelset,
            'f_4d_gaussian': TestFunctions.f_4d_gaussian_pdf
        }
        return function_map[name]


# Test and visualization utilities
def plot_1d_function(func: Callable, x_range: Tuple[float, float] = (-1, 1), 
                      n_points: int = 1000, title: str = ""):
    """Plot 1D test function"""
    x = torch.linspace(x_range[0], x_range[1], n_points)
    y = func(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y.numpy(), 'b-', linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_2d_function(func: Callable, x_range: Tuple[float, float] = (-1, 1),
                     n_points: int = 100, title: str = ""):
    """Plot 2D test function"""
    x1 = torch.linspace(x_range[0], x_range[1], n_points)
    x2 = torch.linspace(x_range[0], x_range[1], n_points)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    
    Z = func(X1, X2)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X1.numpy(), X2.numpy(), Z.numpy(), levels=50, cmap='viridis')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title(f'{title} - Contour')
    plt.colorbar(contour, ax=ax1)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X1.numpy(), X2.numpy(), Z.numpy(), 
                           cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x1,x2)')
    ax2.set_title(f'{title} - Surface')
    
    plt.tight_layout()
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    print("FMMNN Test Functions Implementation")
    print("===================================")
    
    # Test 1D functions
    print("Testing 1D functions...")
    plot_1d_function(TestFunctions.f1_smooth_oscillatory, title="f1: Smooth Oscillatory")
    plot_1d_function(TestFunctions.f2_nonsmooth_oscillatory, title="f2: Non-smooth Oscillatory") 
    plot_1d_function(TestFunctions.f3_nonsmooth_oscillatory, title="f3: Non-smooth Oscillatory")
    
    # Test 2D functions
    print("Testing 2D functions...")
    plot_2d_function(TestFunctions.f_2d_oscillatory_v1, title="2D Oscillatory v1")
    plot_2d_function(TestFunctions.f_2d_oscillatory_v2, title="2D Oscillatory v2")
    
    # Test function access
    f1 = TestFunctions.get_test_function('f1')
    x_test = torch.linspace(-1, 1, 10)
    y_test = f1(x_test)
    print(f"f1 test values shape: {y_test.shape}")
    print("Implementation complete!")