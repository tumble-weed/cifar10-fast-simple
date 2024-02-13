class custom_scheduler1:
    def __init__(self):
        self.lr_schedule = torch.cat([
            torch.linspace(0e+0, 2e-3, 194),
            torch.linspace(2e-3, 2e-4, 584),
        ])
        self.lr_schedule_bias = 64.0 * self.lr_schedule
        self.index = 0

    def lr(self):
        return self.lr_schedule[self.index]
    
    def lr_bias(self):
        return self.lr_schedule_bias[self.index]

    def step(self):
        self.index = min(self.index + 1, len(self.lr_schedule) - 1)

def get_scheduler(scheduler):
    if scheduler == 'custom_scheduler1':
        return custom_scheduler1()
    else:
        dutils.TODO