import torch
from copy import deepcopy

def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


class GFN:
    def __init__(self, config, pretrained_model, use_lora=False, sampler='ddpm'):
        """
        pretrained_model: pretrained diffusion model with he decode sample function
        """
        self.prior = pretrained_model
        self.config = config
        if use_lora:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                target_modules=[]
            )
            self.posterior = get_peft_model(pretrained_model, config)
        else:
            self.posterior = deepcopy(pretrained_model)
        self.posterior = self.posterior.train()
        for param in self.posterior.parameters():
            param.requires_grad = True
        self.prior.eval()
        self.device = next(pretrained_model.parameters()).device
        self.sampler = sampler

    def _ddpm_update_finetune_lp(self, model, x, t, dt, ret_x=None):
        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = model(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]
        if ret_x is None:
            _x = _sample_categorical(q_xs)
            copy_flag = (x != model.mask_index).to(x.dtype)
            ret_x = copy_flag * x + (1 - copy_flag) * _x
        logprobs = torch.distributions.Categorical(probs=q_xs).log_prob(ret_x).sum(-1)
        
        return ret_x, logprobs
    
    def sample_fwd(self, batch_size, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        
        if batch_size is None:
            batch_size_per_gpu = self.config.loader.eval_batch_size
        else:
            batch_size_per_gpu = batch_size
        
        if num_steps is None:
            num_steps = self.config.sampling.steps
        # import pdb; pdb.set_trace();
        total_prior_log_score = torch.zeros(batch_size, num_steps, device=self.device)
        total_posterior_log_score = torch.zeros(batch_size, num_steps, device=self.device)

        with torch.no_grad():
            x = self.prior._sample_prior(
                batch_size_per_gpu,
                self.config.model.length).to(self.device)
        
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                # if get_lp:
                # import pdb; pdb.set_trace();
                x, logprob = self._ddpm_update_finetune_lp(self.posterior, x, t, dt)
                with torch.no_grad():
                    _, logprob_prior = self._ddpm_update_finetune_lp(self.prior, x, t, dt, ret_x=x)
                x = x.detach()
            else:
                raise NotImplementedError("Only DDPM is supported")
            total_posterior_log_score[:, i] = logprob
            total_prior_log_score[:, i] = logprob_prior
        
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                        device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.prior.noise(t)[0]
                logits = self.prior.forward(x, unet_conditioning)
                # x=argmax of logits of the unmasked tokens
                # no issue with subs; for sedd, if not using [:, :, :-1], some samples will contain the mask token
                x = logits[:, :, :-1].argmax(dim=-1)

        return {
            'x': x,
            'posterior_log_probs': total_posterior_log_score,
            'prior_log_probs': total_prior_log_score,
        }
