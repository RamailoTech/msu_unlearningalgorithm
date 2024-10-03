import torch
from torch.optim import Adam
from tqdm import tqdm

class ESDTrainer(UnlearningTrainer):
    """
    ESD-specific implementation of the UnlearningTrainer.
    """

    def __init__(self, model, learning_rate=1e-5, iterations=1000):
        self.model = model
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.optimizer = Adam(self.model.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()

    def train(self, model, sampler, config, **kwargs):
        # for iteration in tqdm(range(self.iterations)):
        #     self.optimizer.zero_grad()
        #     loss = self._compute_loss(prompt, sampler, ddim_steps)
        #     loss.backward()
        #     self.optimizer.step()
        #     print(f"Iteration {iteration + 1}: Loss = {loss.item()}")
        pass

    def _compute_loss(self, prompt, sampler, ddim_steps):
        emb_p = self.model.get_learned_conditioning([prompt])
        start_code = torch.randn((1, 4, 64, 64)).to(self.model.device)
        
        z = sampler.sample(conditioning=emb_p, h=512, w=512, ddim_steps=ddim_steps, scale=1.0, eta=0.0, start_code=start_code)
        e_0 = self.model.apply_model(z, conditioning='')
        e_p = self.model.apply_model(z, conditioning=prompt)

        loss = self.criterion(e_p, e_0)
        return loss
