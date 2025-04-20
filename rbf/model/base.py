import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from rbf.utils.print_utils import print_info

class BaseModel(ABC):
    """
    Abstract base class for all 3D models. This class defines the interface that
    all 3D models must implement.
    """

    def __init__(self) -> None:
        pass

    @property
    def device(self) -> str:
        """
        Returns the device (cpu or cuda) to be used for computations.
        
        :return: Device to be used (cpu or cuda).
        :raises AssertionError: If the configuration is not initialized.
        """
        assert self.cfg is not None, "Configuration not initialized"
        return self.cfg.device

    @property
    def max_steps(self) -> int:
        """
        Returns the maximum number of steps for optimization.
        
        :return: Maximum number of steps.
        :raises AssertionError: If the configuration is not initialized.
        """
        assert self.cfg is not None, "Configuration not initialized"
        return self.cfg.max_steps

    # @abstractmethod
    # def load(self, path: str) -> None:
    #     """
    #     Load the model from the specified path.
        
    #     :param path: Path to the file from which to load the model.
    #     """
    #     pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        :param path: Path to the file to which to save the model.
        """
        pass

    @abstractmethod
    def prepare_optimization(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Prepare the model for optimization. This might include setting up optimizers and other related tasks.
        
        :param parameters: Optional dictionary of parameters for optimization.
        """
        pass

    @abstractmethod
    def render(self, camera) -> torch.Tensor:
        """
        Render the 3D model from the specified camera parameters.
        
        :param camera: Dictionary containing the camera parameters.
        :return: A dictionary containing the rendered images and other information.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        """
        Render the model from its own camera parameters.

        :return: A tensor containing the rendered image.
        """
        pass
    
    @torch.no_grad()
    def render_eval(self, path):
        """
        Render the current model for evaluation.
        """
        print_info("render_eval not implemented.")

    @abstractmethod
    def optimize(self, step: int) -> None:
        """
        Optimize the model parameters for the given step.
        
        :param step: Current step of the optimization process.
        """
        pass

    def regularize(self):
        """
        Regularize the model parameters (optional).
        """
        return torch.tensor(0.0, device=self.device)
    
    def schedule_lr(self, step):
        """
        Adjust the learning rate (optional).
        """
        raise NotImplementedError("Learning rate scheduling not implemented")
    
    @torch.no_grad()
    def closed_form_optimize(self, step, camera, target):
        """
        Perform closed-form optimization for the model parameters (optional).
        """
        raise NotImplementedError("Closed-form optimization not implemented")
    
    def guide_x0(self, step, camera, target):
        """
        Guide the initial guess for optimization (optional).
        """
        raise NotImplementedError("Initial guess guidance not implemented")