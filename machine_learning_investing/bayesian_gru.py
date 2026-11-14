"""Diagonal-Gaussian variational distributions over the two GRU weight matrices."""

import math

import torch
from torch import nn

POSTERIOR_DRAWS = 32
PRIOR_SD = 0.25
INITIAL_SD = 0.03
KL_WEIGHT = 0.02


class VariationalGRU(nn.Module):
    def __init__(self, gru, seed):
        super().__init__()
        self.gru = gru  # Reuse the deterministic GRU's initialization and biases.
        self.input_log_sd = nn.Parameter(torch.full_like(gru.weight_ih_l0, math.log(INITIAL_SD)))
        self.hidden_log_sd = nn.Parameter(torch.full_like(gru.weight_hh_l0, math.log(INITIAL_SD)))
        generator = torch.Generator().manual_seed(seed)
        for name, log_sd in (("input", self.input_log_sd), ("hidden", self.hidden_log_sd)):
            noise = torch.randn(POSTERIOR_DRAWS, *log_sd.shape, generator=generator)
            self.register_buffer(f"{name}_draws", noise)

    def weight_terms(self):
        return ((self.gru.weight_ih_l0, self.input_log_sd),
                (self.gru.weight_hh_l0, self.hidden_log_sd))

    def kl_per_weight(self):
        # Analytic KL of the learned factorized Gaussian against N(0, PRIOR_SD**2).
        total = 0.0
        count = 0
        for mean, log_sd in self.weight_terms():
            variance = log_sd.exp().square()
            quadratic = (mean.square() + variance) / (2 * PRIOR_SD**2)
            kl = math.log(PRIOR_SD) - log_sd + quadratic - 0.5
            total = total + kl.sum()
            count += mean.numel()
        return total / count

    def final_states(self, inputs, input_weights, hidden_weights):
        # The sample axis stays fixed throughout the entire observation window.
        # PyTorch's GRU gate order is reset, update, new.
        draws = input_weights.shape[0]
        batch, steps, _ = inputs.shape
        gates = inputs.flatten(0, 1) @ input_weights.transpose(-2, -1)
        gates = (gates + self.gru.bias_ih_l0).reshape(draws, batch, steps, -1)
        hidden = inputs.new_zeros(draws, batch, self.gru.hidden_size)
        recurrent = hidden_weights.transpose(-2, -1)
        for step in gates.unbind(dim=2):
            x_reset, x_update, x_new = step.chunk(3, dim=-1)
            h_reset, h_update, h_new = (hidden @ recurrent + self.gru.bias_hh_l0).chunk(3, dim=-1)
            reset = (x_reset + h_reset).sigmoid()
            update = (x_update + h_update).sigmoid()
            candidate = (x_new + reset * h_new).tanh()
            hidden = torch.lerp(candidate, hidden, update)
        return hidden

    def forward(self, inputs):
        # One reparameterized sample per optimizer step.
        weights = [mean + log_sd.exp() * torch.randn_like(mean)
                   for mean, log_sd in self.weight_terms()]
        return self.final_states(inputs, *(w.unsqueeze(0) for w in weights))[0]

    def posterior_states(self, inputs):
        weights = [mean + log_sd.exp() * noise for (mean, log_sd), noise
                   in zip(self.weight_terms(), (self.input_draws, self.hidden_draws))]
        return self.final_states(inputs, *weights)
