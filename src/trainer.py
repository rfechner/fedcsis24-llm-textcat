
from transformers.trainer import Trainer
from torch._tensor import Tensor
from collections import Counter
from torch.nn import CrossEntropyLoss

class ImbaTrainer(Trainer):
    """
        Using Class Balanced Loss: https://arxiv.org/pdf/1901.05555.pdf

        We assume that to re-weight the loss we have to respect the fact that
        the more samples a class has, the more data-redundancy is introduced.

        The proposed scheme re-balances the loss under the assumption that
        additional samples have deminishing information about the class. The authors
        propose to calculate the effective number E_i for every class i, which is the inverse scaling
        factor applied to the loss.
        E_i(n) = (1 - beta^n) / (1 - beta) is a function dependent on the sample frequency for class i, growing linearly for small
        frequencies, then saturating for larger sample counts. The function goes to 1 / (1 - beta) where
        beta is chosen to be (N-1)/N where N is the total number of samples.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_dataset = kwargs['train_dataset']
        self.counts = Counter(train_dataset['label'])
        self.num_labels = len(self.counts.keys())
        self.rebalance_N = len(train_dataset['label'])
        self.rebalance_beta = (self.rebalance_N - 1.0) / self.rebalance_N
        self.inv_effective_number = (1.0 - self.rebalance_beta) / (1.0 - self.rebalance_beta**Tensor(list(self.counts.values())))
        
    def compute_loss(self, model, inputs, return_outputs=False):  
        outputs = model(**inputs)
        logits = outputs[1]
        loss_fct = CrossEntropyLoss(weight=self.inv_effective_number.to(model.device))
        loss = loss_fct(logits.view(-1, self.num_labels), inputs['labels'].view(-1))
        
        # have to return like this, on the evaluate loop call, we need to return (loss, outputs)
        return (loss, outputs) if return_outputs else loss
    
    
