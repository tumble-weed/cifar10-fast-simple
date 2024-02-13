class nesterov:
    def __init__(self, weight_decay, weight_decay_bias, momentum, ema_update_freq, rho):
        self.weight_decay = weight_decay
        self.weight_decay_bias = weight_decay_bias
        self.momentum = momentum
        self.ema_update_freq = ema_update_freq
        self.rho = rho
    
    def update_nesterov(self, weights, lr, weight_decay, momentum):
        for weight, velocity in weights:
            if weight.requires_grad:
                gradient = weight.grad.data
                weight = weight.data

                gradient.add_(weight, alpha=weight_decay).mul_(-lr)
                velocity.mul_(momentum).add_(gradient)
                weight.add_(gradient.add_(velocity, alpha=momentum))
    
    def update_ema(self, train_model, valid_model, rho):
    # The trained model is not used for validation directly. Instead, the
    # validation model weights are updated with exponential moving averages.
        train_weights = train_model.state_dict().values()
        valid_weights = valid_model.state_dict().values()
        for train_weight, valid_weight in zip(train_weights, valid_weights):
            if valid_weight.dtype in [torch.float16, torch.float32]:
                valid_weight *= rho
                valid_weight += (1 - rho) * train_weight

    def step(self, weights, biases, scheduler, train_model, valid_model):
        self.update_nesterov(weights, scheduler.lr(), self.weight_decay, self.momentum)
        self.update_nesterov(biases, scheduler.lr_bias(), self.weight_decay_bias, self.momentum)
        self.update_ema(train_model, valid_model, self.rho)

def get_optimizer(optimizer, weight_decay, weight_decay_bias, momentum, ema_update_freq, rho):
    if optimizer == 'nesterov':
        return nesterov(weight_decay, weight_decay_bias, momentum, ema_update_freq, rho)
    else:
        dutils.TODO