import torch
import torch.nn as nn
import torch.nn.functional as F

"""

Attention-based Deep Multiple Instance Learning Model.

Implementation influenced by Ilse et al. (2018).

"""

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 512 #embedding size for bag
        self.D = 128 #hidden embedding size for bag features
        self.K = 1 #number of classes, 1 for breast cancer detection

        self.feature_extractor_part1 = nn.Sequential(
        nn.Conv2d(3, 20, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        #nn.AdaptiveAvgPool2d(output_size=2),
        nn.Conv2d(20, 50, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        #nn.AdaptiveAvgPool2d(output_size=2)
            )

        self.feature_extractor_part2 = nn.Sequential(
        nn.Linear(50 * 29 * 29, self.L), #configuered for input patch size of 128 x 128.
        #nn.Linear(50 * 4 * 4, self.L),
        nn.ReLU(),
        #nn.Dropout(0.5)
            )

        self.attention = nn.Sequential(
        nn.Linear(self.L, self.D),
        nn.Tanh(),
        nn.Dropout(), #added an instance dropout layer to attention mechanism.
        nn.Linear(self.D, self.K)

            )


            # consider adding gated mechanism


        self.classifier = nn.Sequential(
        nn.Linear(self.L*self.K, 1),
        #nn.ReLU(), 
        #nn.Dropout(),
        nn.Sigmoid()
            )

    def forward(self, x):
        #print(x.size())
        #x = x.view(x.size(0),50*22*22) #flatten to do matrix multiplication
        x = x.squeeze(0) # reshapes from 5-dimension input to 4-dimension input [batch_size, color, height, width]
        #x = x.view(x.size(0),-1)
        #print(x.size())
        H = self.feature_extractor_part1(x) 
        #print(H.size())
        #H = H.view(-1, 50 * 4 * 4) #flatten to do matrix multiplication
        H = H.view(H.size(0),-1) #flatten to do matrix multiplication
        #print(H.size())
        H = self.feature_extractor_part2(H)  # NxL
        #print(H.size())

        A = self.attention(H)  # NxK
        #print(A.size())
        A = torch.transpose(A, 1, 0)  # KxN
        #print(A.size())
        A = F.softmax(A, dim=1)  # softmax over N
        #print(A.size())

        M = torch.mm(A, H)  # KxL
        #print(M.size())

        Y_prob = self.classifier(M)
        #print(Y_prob)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        #print(Y_hat)

        return Y_prob, Y_hat, A 
        #print("Probability": Y_prob)
        #print("Attention": A)

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
