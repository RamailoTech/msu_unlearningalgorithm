# mu_attack/core/base_attack.py

from abc import ABC, abstractmethod

class Attacker(ABC):
    def __init__(
        self,
        iteration,
        seed_iteration,
        insertion_location,
        k,
        eval_seed,
        attack_idx,
        universal,
        sequential,
        **kwargs 
    ):
        self.iteration = iteration
        self.seed_iteration = seed_iteration
        self.insertion_location = insertion_location
        self.k = k
        self.eval_seed = eval_seed
        self.universal = universal
        self.attack_idx = attack_idx
        self.sequential = sequential


        for key, value in kwargs.items():
            if not hasattr(self, key):  
                setattr(self, key, value)

    @abstractmethod
    def run(self):
        """
        Abstract method that must be implemented by subclasses.
        """
        pass
