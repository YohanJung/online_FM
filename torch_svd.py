import numpy as np
import torch

tensor_type = torch.DoubleTensor

def grad(x):
    # Alternatively: https://math.stackexchange.com/a/934443
    # from scipy.linalg import sqrtm
    # return sqrtm(np.linalg.inv(x @ x.transpose())) @ x

    # Using SVD: https://math.stackexchange.com/a/701104 + https://math.stackexchange.com/a/1663012
    u, sig, v = torch.svd(x)
    return u.matmul(v.t()).double()


if __name__ == "__main__" :

    x = torch.rand(10, 5).double()
    x /= torch.norm(x)


    for i in range(10000):
        if i % 100 == 0:
            print("Nuclear norm =", np.linalg.norm(x.numpy(), 'nuc'), "; k =", np.linalg.cond(x.numpy()))
            x -= 0.0001 * grad(x)

        x /= torch.norm(x)


    print(x)




