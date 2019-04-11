import numpy as np

#np.zeros([3,1]).double()

def init_fm( i , n_fm):
    if i not in n_fm:
        n_fm[i] = np.zeros(10, dtype=np.float64)

    return n_fm


if __name__ == "__main__":
    # n_fm = {}
    # n_fm = init_fm(3,n_fm)
    # print(n_fm)
    #
    # n_fm = init_fm(5,n_fm)
    # print(n_fm)
    #
    # n_fm = init_fm(10,n_fm)
    # print(n_fm)
    #
    # n_fm = init_fm(3,n_fm)
    # print(n_fm)
    #
    # print(np.zeros(5))
    #
    # print(np.random.randn())
    #
    # print(  1/ (1+ np.exp(-40)) )

    # x = np.zeros(10)
    # x[1] = 1
    # x[6] = 1
    # print(x.shape)
    #
    # for x_ith in x:
    #     print(x_ith)
    #     #print('')

    for i in range(10):
        if 1:
            print('')


    x = np.zeros(10)
    print(x.size)