# BehaviorMemoryBlender.py
# Evolves Friday’s personality profile over time based on emotional memory and tone logs

class BehaviorMemoryBlender:
    def __init__(self, alpha: float = 0.1):
        """
        :param alpha: blending factor (0 = ignore new, 1 = overwrite old)
        """
        self.alpha = alpha  # learning rate for gradual change

    def blend_traits(self, old_traits: dict, recent_averages: dict) -> dict:
        """
        Blend the old personality traits with recent tone averages.
        """
        blended = {}
        for trait in old_traits:
            old_val = old_traits.get(trait, 0.5)
            new_val = recent_averages.get(trait, 0.5)
            blended[trait] = round((1 - self.alpha) * old_val + self.alpha * new_val, 4)
        return blended
