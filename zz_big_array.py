import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, load_npz, save_npz

import zz_sparse_corr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class elapsed_tick:
    def __init__(self):
        self.start = time.time()
        return

    @property
    def elapsed(self):
        ret = time.time()-self.start
        ret = np.round(ret, 6)
        return ret


class big_array:
    '''
    '''

    def __init__(self, search_index):
        '''
        '''
        self.search_index = search_index
        return

    def run(self):
        '''
        '''
        tick = elapsed_tick()

        self.process_memmap()

        print(f'[run]:needs ({tick.elapsed}) secs.')
        return

    def process_memmap(self):
        '''
        '''
        print(f'[process_memmap].')

        total_ptime = 0

        tick = elapsed_tick()

        # rows,cols=30000,30000
        rows, cols = 20000, 20000
        # rows,cols=10000,10000

        # A=np.memmap('big_matrix.dat', dtype=np.int32,mode='w+',shape=(rows,cols))
        # A[:]=np.random.random((rows,cols))

        name = "microns"
        A = load_npz(f"{name}_allW.npz")
        print(f"Density:{np.sum(A)/(A.shape[0]**2)}")
        print(A.data.nbytes/(1024*1024*1024))
        rows, cols = A.shape[0], A.shape[1]

        total_ptime += tick.elapsed
        print(f'[process_memmap]:create big_matrix needs ({tick.elapsed}) secs.')

        total_rows = A.shape[0]
        # better for memory
        B = np.memmap(f'corr_W_{name}_{self.search_index}.dat',
                      dtype=np.float64, mode='w+', shape=(rows, cols))
        # by default
        # B=np.empty((cols,cols))

        ptime = 0
        sample_cnt = 10
        break_pt = 100
        print(f"break_pt: {break_pt}")

        for i in range(0, total_rows, break_pt):
            # if i>=sample_cnt:
            #   print(f'[process_memmap]:beyond sample_cnt({sample_cnt}).')
            #   break
            tick = elapsed_tick()
            print(f'[process_memmap]:process row({i}/{total_rows})', end=' ')

            if self.search_index == "row":
                ra = A[i:i+break_pt, :]
            elif self.search_index == "column":
                ra = A[:, i:i+break_pt].T

            for j in range(0, total_rows, break_pt):
                if self.search_index == "row":
                    rb = A[j:j+break_pt:]
                elif self.search_index == "column":
                    rb = A[:, j:j+break_pt].T

                # if not isinstance(ra, np.ndarray):
                #     ra_new=ra.toarray()
                # if not isinstance(rb, np.ndarray):
                #     rb_new=rb.toarray()

                # dot2=np.corrcoef(ra_new,rb_new)

                dot = zz_sparse_corr.sparse_pearson_correlation_N_on_N(ra, rb)

                # test=np.zeros((break_pt,break_pt))
                # for ind1 in range(break_pt):
                #     for ind2 in range(break_pt):
                #         # test[ind1,ind2]=zz_sparse_corr.sparse_pearson_correlation_1_on_1(ra[ind1,:],rb[ind2,:])
                #         corr = np.corrcoef(ra[ind1,:].toarray().flatten(),rb[ind2,:].toarray().flatten())[0,1]
                #         test[ind1,ind2]=corr

                # print(dot-test)

                B[i:i+break_pt, j:j+break_pt] = dot
                
            ptime += tick.elapsed
            print(f'needs ({tick.elapsed}) secs.')

        ptime = (ptime/sample_cnt)*total_rows

        total_ptime += ptime

        print(f'[process_memmap]:evaluate process time ({total_ptime}) secs.')

        # 关闭big_matrix时确保调用big_matrix.flush() 和 big_matrix.close()
        B.flush()
        del B

        print(f'[process_memmap]:done.')

        return


if __name__ == '__main__':
    search_index = "column"
    ba = big_array(search_index)
    ba.run()

    # corr_W = np.fromfile('corr_W.dat', dtype=np.float64)
    # corr_W = corr_W.reshape(int(np.sqrt(corr_W.shape[0])), int(np.sqrt(corr_W.shape[0])))
    # print(corr_W)

    # np.fill_diagonal(corr_W, 0)
    # downsample_factor = 4
    # corr_W = corr_W[::downsample_factor, ::downsample_factor]

    # plt.figure()
    # sns.heatmap(corr_W+1e-10, cbar=True, square=True, norm=LogNorm())
    # plt.savefig("corr_W.png")

    exit()
