"""CASCADE model components: CCMP backbone, propagation head, full model.

The backbone needs torch. Imports are guarded so ``import cascade.models``
still succeeds in a torch-free environment (for example the lightweight CI
image used for the data and features layers); the symbols simply are not
exported there.
"""

try:
    from .cascade import CASCADE
    from .ccmp_backbone import CCMPBackbone, ConceptStream
    from .concept_gnn import RNAFM_CONCEPT, ConceptGNN
    from .propagation_head import PropagationHead
    from .rc_equivariant import RCEquivariantLinear

    __all__ = [
        "CASCADE",
        "CCMPBackbone",
        "ConceptGNN",
        "ConceptStream",
        "PropagationHead",
        "RCEquivariantLinear",
        "RNAFM_CONCEPT",
    ]
except ImportError:  # torch not installed
    __all__ = []
