"""Logical low-bit payload adapter; actual bit packing/networking not implemented."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import torch
from compression.quantization import stochastic_k_level_quantize
from compression.adaptive_quantization import adaptive_quantize

METHODS = ('minmax-stochastic','minmax-nearest','optimized-uniform','lloyd-max',
           'fixed-uniform','fixed-lloyd-max')


def canonical(method, bits):
    if bits == 1:
        return {'lloyd-max':'optimized-uniform','fixed-lloyd-max':'fixed-uniform'}.get(method,method)
    return method


def checked_values(values):
    if not values.is_floating_point() or not values.numel() or not torch.isfinite(values).all():
        raise ValueError('Expected nonempty finite floating-point coefficients')
    x = values.float()
    if not torch.isfinite(x).all() or torch.any((values != 0) & (x == 0)):
        raise ValueError('Coefficients cannot be represented with float32 payload metadata')
    return x


def rms32(values):
    # Double accumulators avoid overflow/underflow from squaring float32 values.
    scale = values.double().square().mean().sqrt().float()
    if not torch.isfinite(scale) or (scale == 0 and torch.any(values != 0)):
        raise ValueError('Nonzero RMS is outside representable float32 scale range')
    return scale


def uniform_centers(metadata, bits):
    t = torch.arange(2**bits,device=metadata.device,dtype=torch.float64)/(2**bits-1)
    return (metadata[0].double()+t*(metadata[1].double()-metadata[0].double())).float()


def nearest_indices(values, centers):
    # Midpoint computation must not overflow when finite endpoints are large.
    boundaries=(centers[:-1].double()+centers[1:].double())/2
    return torch.bucketize(values.double().contiguous(),boundaries)


@dataclass
class Payload:
    indices: torch.Tensor
    metadata: torch.Tensor
    bits: int
    method: str
    representation: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def wire_bits(self):
        return self.indices.numel()*self.bits + self.metadata.numel()*32


class CodebookRegistry:
    """Shared offline artifact, containing method-specific standardized centers."""
    def __init__(self, codebooks, provenance=None):
        self._centers = {key:torch.as_tensor(value,dtype=torch.float32).clone() for key,value in codebooks.items()}
        for value in self._centers.values():
            if value.ndim != 1 or not torch.isfinite(value).all() or torch.any(value[1:] < value[:-1]):
                raise ValueError('Invalid codebook')
        self.provenance = provenance or {}

    @staticmethod
    def key(representation,bits,method):
        return f'{representation}:{bits}:{canonical(method,bits)}'

    def centers(self,representation,bits,method,device='cpu'):
        result = self._centers[self.key(representation,bits,method)]
        if len(result) != 2**bits:
            raise ValueError('Codebook size does not match bit width')
        return result.to(device).clone()

    @property
    def shared_center_bits(self):
        return sum(value.numel()*32 for value in self._centers.values())

    def document(self):
        return dict(schema=1,normalization='uncentered RMS; float32 transmitted scale',
                    codebooks={k:v.tolist() for k,v in self._centers.items()},provenance=self.provenance,
                    shared_center_bits=self.shared_center_bits,
                    sharing='preinstalled for these experiments; center distribution cost listed separately')

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.document(),sort_keys=True).encode()).hexdigest()

    def save(self,path):
        document = self.document()
        document['sha256'] = self.fingerprint()
        Path(path).write_text(json.dumps(document,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        obj = cls(doc['codebooks'],doc['provenance'])
        if doc['schema'] != 1 or doc['sha256'] != obj.fingerprint():
            raise ValueError('Codebook artifact hash/schema mismatch')
        return obj


@torch.no_grad()
def encode(values,bits,method,representation,registry=None,generator=None):
    if bits not in (1,2) or method not in METHODS:
        raise ValueError('Unsupported quantizer/bit width')
    x = checked_values(values)
    diagnostics = {}
    if method.startswith('fixed-'):
        if registry is None:
            raise ValueError('Fixed quantization requires a frozen registry')
        scale = rms32(x)
        metadata = scale.reshape(1).clone()
        centers = registry.centers(representation,bits,method,x.device)
        normalized = x/scale if scale > 0 else torch.zeros_like(x)
        idx = nearest_indices(normalized,centers)
        diagnostic_values = normalized
        diagnostics.update(fit='offline-frozen',scale=float(scale))
    elif method in ('minmax-stochastic','minmax-nearest'):
        metadata = torch.stack([x.min(),x.max()])
        centers = uniform_centers(metadata,bits)
        if method == 'minmax-stochastic':
            _,idx = stochastic_k_level_quantize(x,2**bits,generator)
        else:
            idx = nearest_indices(x,centers)
        diagnostic_values = x
        diagnostics.update(fit='none')
    else:
        _,idx,metadata,_ = adaptive_quantize(x,bits,method,diagnostics=diagnostics)
        if not torch.isfinite(metadata).all():
            raise ValueError('Per-vector fit exceeds float32 numeric range')
        centers = uniform_centers(metadata,bits) if method == 'optimized-uniform' else metadata
        # Assign using the actual transmitted codebook for exact nearest decoding.
        idx = nearest_indices(x,centers)
        diagnostic_values = x
        diagnostics.update(fit='per-vector-local')
    diagnostics['outside_extreme_fraction'] = float(((diagnostic_values<centers[0]) |
                                                     (diagnostic_values>centers[-1])).float().mean())
    return Payload(idx.long(),metadata.float().clone(),bits,method,representation,diagnostics)


@torch.no_grad()
def decode(payload,registry=None):
    """Only indices, transmitted float32 metadata, and optional shared registry."""
    if payload.bits not in (1,2) or payload.method not in METHODS:
        raise ValueError('Unsupported payload')
    idx,meta = payload.indices,payload.metadata
    if idx.dtype != torch.int64 or not idx.numel() or torch.any(idx<0) or torch.any(idx>=2**payload.bits):
        raise ValueError('Invalid or out-of-range indices')
    if meta.ndim != 1 or meta.device != idx.device or meta.dtype != torch.float32 or not torch.isfinite(meta).all():
        raise ValueError('Metadata must be finite float32')
    if payload.method.startswith('fixed-'):
        if meta.numel()!=1 or meta[0]<0 or registry is None:
            raise ValueError('Invalid fixed payload/registry')
        centers = registry.centers(payload.representation,payload.bits,payload.method,idx.device)
        result = meta[0]*centers[idx]
    elif payload.method == 'lloyd-max':
        if meta.numel()!=2**payload.bits or torch.any(meta[1:]<meta[:-1]):
            raise ValueError('Invalid Lloyd-Max metadata')
        result = meta[idx]
    else:
        if meta.numel()!=2 or meta[0]>meta[1]:
            raise ValueError('Invalid uniform endpoints')
        result = uniform_centers(meta,payload.bits)[idx]
    if not torch.isfinite(result).all():
        raise ValueError('Decoded coefficients overflow float32')
    return result
