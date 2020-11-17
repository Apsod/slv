import copy
import torch
import transformers

def mean_pooled(model, data):
    ixs, att = data
    outs, *_ = model(input_ids=ixs, attention_mask=att)

    mean_mask = torch.true_divide(
            att,
            att.sum(1, keepdims=True)
        )

    return (outs * mean_mask.unsqueeze(2)).sum(1)


class QA_model(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        model = transformers.AutoModel.from_pretrained(architecture)
        self.left = model
        self.right = copy.deepcopy(model)

    def forward(self, data):
        """
        Takes data in the form (token_idx, attention_mask) where
        token_idx, attention_mask both have shape (B x L),
        i.e. Batch size times sequence lenth.
        Returns two (B x D) tensors ("left" and "right")
        Any function overriding forward should be
        semantically identical to this one.
        """
        return self.generate_left(data), self.generate_right(data)

    def generate_left(self, data):
        """
        Takes data in the form (token_idx, attention_mask) where
        token_idx, attention_mask both have shape (B x L),
        i.e. Batch size times sequence lenth.
        Returns the "left" (B x D) tensors.
        """
        return mean_pooled(self.left, data)

    def generate_right(self, data):
        """
        Takes data in the form (token_idx, attention_mask) where
        token_idx, attention_mask both have shape (B x L),
        i.e. Batch size times sequence lenth.
        Returns the "right" (B x D) tensors.
        """
        return mean_pooled(self.right, data)

    def loss(self, q, a, loss='bce'):
        # q_emb, a_emb : B x D
        q_emb = self.generate_left(q)
        a_emb = self.generate_right(a)

        B, D = q_emb.shape

        scores = q_emb @ a_emb.t()

        if loss == 'bce':
            targets = torch.eye(B, device=scores.device)
            l = torch.nn.functional.binary_cross_entropy_with_logits(scores, targets)
        elif loss == 'ce':
            targets = torch.arange(B, dtype=torch.long, device=scores.device)
            l = torch.nn.functional.cross_entropy(scores, targets)
        return l

    def save(self, path):
        torch.save((self.state_dict(), self.architecture), path)

    @classmethod
    def load(cls, path):
        state_dict, architecture = torch.load(path, map_location='cpu')
        model, tok = cls.init(architecture)
        model.load_state_dict(state_dict)
        return model, tok

    @classmethod
    def init(cls, architecture):
        """
        Initialize CT model from pretrained huggingface model.
        """
        tok = transformers.AutoTokenizer.from_pretrained(architecture)
        return cls(architecture), tok

