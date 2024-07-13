from typing import Iterable
import warnings
import torch
from torchvision.models.vision_transformer import VisionTransformer, Encoder
from torchvision.models.resnet import ResNet
from torchvision.models.vgg import VGG
from torchvision.models.swin_transformer import SwinTransformer
import fasterdp
import contextvars

context_var_grad_accum_step = contextvars.ContextVar('grad_accum_step', default=-1)

class StellaTrainDataLoader():
    def __init__(self, iterable: Iterable):
        self._iterable = iterable

    def __iter__(self):
        # read configuration from fasterdp
        grad_acc: int = fasterdp.gradient_accumulation()

        for obj in self._iterable:
            # traverse inside obj. If tensor, chunk into grad_acc. If list or tuple, traverse

            def traverse(obj, i):
                if isinstance(obj, torch.Tensor):
                    # chunk tensor
                    return torch.chunk(obj, grad_acc, dim=0)[i]
                elif isinstance(obj, list):
                    # traverse list or tuple
                    return list(traverse(o, i) for o in obj)
                elif isinstance(obj, tuple):
                    # traverse list or tuple
                    return tuple(traverse(o, i) for o in obj)
                elif isinstance(obj, dict):
                    raise NotImplementedError()
                else:
                    # return obj
                    return obj
            
            # this is inefficient operation, as calling chunk multiple times
            # but will do for now
            objs = [traverse(obj, i) for i in range(grad_acc)]

            for i, inner_obj in enumerate(objs):
                context_var_grad_accum_step.set(i)
                yield inner_obj

            context_var_grad_accum_step.set(-1)

    @property
    def is_eoi(self) -> bool:
        '''
        Whether current micro-iteration is the end of iteration
        '''
        grad_acc: int = fasterdp.gradient_accumulation()
        grad_acc_step: int = context_var_grad_accum_step.get()

        assert grad_acc_step >= 0
        
        # print(f"Iterating {grad_acc_step} / {grad_acc}")
        return grad_acc_step == grad_acc - 1
    
class FasterDpModelWrapper(torch.nn.Module):

    def tensor_forward_hook(self, module: torch.nn.Module, layer_idx: int = -1) -> None:
        for name, gpu_param in module.named_parameters():
            assert gpu_param.requires_grad
            fasterdp.pre_forward_process(layer_idx, name)

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = self.to_sequential(module)
        self.is_initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # read configuration from fasterdp
        grad_acc_step: int = context_var_grad_accum_step.get()
        assert grad_acc_step >= 0, "Need to use StellaTrainDataLoader"
        
        if not self.is_initialized:
            self.is_initialized = True
            for layer_idx, layer in enumerate(self.module):
                for name, gpu_param in layer.named_parameters():
                    assert gpu_param.is_contiguous()
                    fasterdp.pre_train_init(layer_idx, name, gpu_param)

        if grad_acc_step > 0:
            # this micro-iteration is not the beginning of the iteration
            # do not wait for model update
            # print("not start")
            x = self.module(x)
        else:
            # this micro-iteration is the beginning of iteration.
            # wait for model update
            # print("start")
            for layer_idx, layer in enumerate(self.module):
                self.tensor_forward_hook(layer, layer_idx)
                x = layer(x)

        return x

    def step(self):
        # if DEBUG_ACCURACY is enabled, force sync model (dry_run) to check synchronized status
        for layer_idx, layer in enumerate(self.module):
            for name, _ in layer.named_parameters():
                fasterdp.force_model_sync(layer_idx, name, True)

    def zero_grad(self, set_to_none: bool = False) -> None:
        warnings.warn("zero_grad is not supported in FasterDpModelWrapper. Gradients are automatically zeroed out when necessary.")        


    @staticmethod
    def to_sequential(module: torch.nn.Module):
        if isinstance(module, torch.nn.Sequential):
            return module

        if isinstance(module, VisionTransformer):
            assert isinstance(module.encoder.layers, torch.nn.Sequential)
            assert isinstance(module.heads, torch.nn.Sequential)
            layers = [
                ViTProcessInput(module),
                ViTPreEncoder(module.encoder),
                module.encoder.dropout,
                *[layer for layer in module.encoder.layers],
                module.encoder.ln,
                ViTIntermediateConvert(),
                *[layer for layer in module.heads],
            ]
            return torch.nn.Sequential(*layers)

        if isinstance(module, ResNet):
            assert isinstance(module.layer1, torch.nn.Sequential)
            assert isinstance(module.layer2, torch.nn.Sequential)
            assert isinstance(module.layer3, torch.nn.Sequential)
            assert isinstance(module.layer4, torch.nn.Sequential)
            layers = [
                module.conv1,
                module.bn1,
                module.relu,
                module.maxpool,
                *[layer for layer in module.layer1],
                *[layer for layer in module.layer2],
                *[layer for layer in module.layer3],
                *[layer for layer in module.layer4],
                module.avgpool,
                torch.nn.Flatten(1),
                module.fc
            ]
            return torch.nn.Sequential(*layers)

        if isinstance(module, SwinTransformer):
            assert isinstance(module.features, torch.nn.Sequential)
            layers = [
                *[layer for layer in module.features],
                module.norm,
                SwinTransformerPermute(),
                module.avgpool,
                torch.nn.Flatten(1),
                module.head
            ]
            return torch.nn.Sequential(*layers)

        if isinstance(module, VGG):
            assert isinstance(module.features, torch.nn.Sequential)
            assert isinstance(module.classifier, torch.nn.Sequential)

            layers = [
                *[layer for layer in module.features],
                module.avgpool,
                torch.nn.Flatten(1),
                *[layer for layer in module.classifier]
            ]
            return torch.nn.Sequential(*layers)
        if module.__class__.__name__ == "GPT":
            layers = [
                GPTPreEncoder(module),
                *[GPTEncoderWrapper(layer) for layer in module.transformer.h],
                GPTEncoderWrapper(module.transformer.ln_f),
                GPTPostEncoder(module)
            ]
            return torch.nn.Sequential(*layers)

        raise NotImplementedError(f"{module.__class__.__name__ } is not implemented")

class SwinTransformerPermute(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

# ViT
class ViTProcessInput(torch.nn.Module):
    def __init__(self, net: VisionTransformer):
        super().__init__()
        self.patch_size = net.patch_size
        self.image_size = net.image_size
        self.conv_proj = net.conv_proj
        self.hidden_dim = net.hidden_dim
        self.class_token = net.class_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        return x

class ViTPreEncoder(torch.nn.Module):
    def __init__(self, net: Encoder):
        super().__init__()
        self.pos_embedding = net.pos_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class ViTIntermediateConvert(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

class DebugWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class GPTPreEncoder(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.transformer_wte = net.transformer.wte
        self.transformer_wpe = net.transformer.wpe
        self.transformer_drop = net.transformer.drop
        self.config = net.config
    def forward(self, x):
        idx, targets = x
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # forward the GPT model itself
        tok_emb = self.transformer_wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer_wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer_drop(tok_emb + pos_emb)
        
        return (x, targets)

class GPTEncoderWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net
    
    def forward(self, x1) -> torch.Tensor:
        x, targets = x1
        return self.net(x), targets
    
from torch.nn import functional as F
class GPTPostEncoder(torch.nn.Module):
    
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.lm_head = net.lm_head
        
    def forward(self, x1):
        x, targets = x1
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss