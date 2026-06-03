"""CASCADE model components: CCMP backbone, propagation head, full model.

The backbone needs torch. Imports are guarded so ``import cascade.models``
still succeeds in a torch-free environment (for example the lightweight CI
image used for the data and features layers); the symbols simply are not
exported there.
"""

try:
    from .cascade import CASCADE
    from .ccmp_backbone import CCMPBackbone, ConceptStream
    from .propagation_head import PropagationHead

    __all__ = ["CASCADE", "CCMPBackbone", "ConceptStream", "PropagationHead"]
except ImportError:  # torch not installed
    __all__ = []
