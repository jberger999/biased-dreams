import torch

nn = torch.nn
jit = torch.jit

        
class _AbstractJittedTD(jit.ScriptModule):
    
    def __init__(self,
                 base_module: nn.Module):
        super(_AbstractJittedTD, self).__init__()
        self._base_module = base_module
        self._copy_through = getattr(self._base_module, "td_copy_trough", None)
        
    @staticmethod
    def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, seq_length = x.shape[:2]
        bs = batch_size * seq_length
        new_shape = [bs, x.shape[2]] if len(x.shape) == 3 else [bs, x.shape[2], x.shape[3], x.shape[4]]
        return x.reshape(new_shape), batch_size, seq_length

    @staticmethod
    def _unflatten(x: torch.Tensor, batch_size: int, seq_length: int) -> torch.Tensor:
        if len(x.shape) == 2:
            new_shape = [batch_size, seq_length, x.shape[1]]
        else:
            new_shape = [batch_size, seq_length, x.shape[1], x.shape[2], x.shape[3]]
        return x.reshape(new_shape)
        
    def forward(self,
                x: torch.Tensor):
        return self._forward(x=x)

    
class Jitted11TD(_AbstractJittedTD):
    
    @jit.script_method
    def _forward(self, x: torch.Tensor):
        x_flat, batch_size, seq_length = self._flatten(x)
        y_flat = self._base_module(x_flat)
        return self._unflatten(y_flat, batch_size, seq_length)
    

class Jitted12TD(_AbstractJittedTD):
    
    @jit.script_method
    def _forward(self, x: torch.Tensor):
        x_flat, batch_size, seq_length = self._flatten(x)
        y1_flat, y2_flat = self._base_module(x_flat)
        y1 = y1_flat if 0 in self._copy_through else self._unflatten(y1_flat, batch_size, seq_length)
        y2 = y2_flat if 1 in self._copy_through else self._unflatten(y2_flat, batch_size, seq_length)
        return y1, y2