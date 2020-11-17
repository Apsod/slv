

def mk_batch2device(device):

    def to_device(batch):
        q, a = batch
        q_i, q_a = q
        a_i, a_a = a
        return ((q_i.to(device), q_a.to(device)),
                (a_i.to(device), a_a.to(device)))

    return to_device


class ExpMean(object):
    def __init__(self, mean=0.0, alpha=0.95):
        self.mean = mean
        self.alpha = alpha

    def __iadd__(self, other):
        self.mean += (other - self.mean) * self.alpha
        return self


class WelfordMean(object):
    def __init__(self, mean=0.0, weight=0.0):
        self.mean = mean
        self.weight = weight

    def __iadd__(self, other):
        if type(other) is WelfordMean:
            self.weight += other.weight
            self.mean += (other.mean - self.mean) * other.weight / self.weight
        else:
            self.weight += 1
            self.mean += (other - self.mean) / self.weight
        return self

