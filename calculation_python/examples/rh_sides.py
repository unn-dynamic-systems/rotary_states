from numba import njit
import numpy as np
import math as mt

class R_SIDES:
    @staticmethod
    @njit
    def coupled_pendulums_rs(q, _, N, L, G, K):
        X = np.empty(2 * N)
        X[0] = q[1]
        X[1] = -L * q[1] - mt.sin(q[0]) + G + K * (mt.sin(q[2] - q[0]))
        n = 0
        while n < 2 * N - 4: 
            X[n + 2] = q[n + 3]
            X[n + 3] = -L * q[n + 3] - mt.sin(q[n + 2]) + G + K * \
                (mt.sin(q[n + 4] - q[n + 2]) + mt.sin(q[n] - q[n + 2]))
            n += 2
        X[2 * N - 2] = q[2 * N - 1]
        X[2 * N - 1] = -L * q[2 * N - 1] - mt.sin(q[2 * N - 2]) + G  \
                + K * (mt.sin(q[2 * N - 4] - q[2 * N - 2]))
        return X

    @staticmethod
    @njit
    def coupled_pendulums_linear_rs(q, _, N, L, K, DSomeRotMode_t):
        X = np.empty(2 * N)
        X[0] = q[1]
        X[1] = -L*q[1] - mt.cos(DSomeRotMode_t[0])*q[0] + K*mt.cos(DSomeRotMode_t[1]-DSomeRotMode_t[0])*(q[2] - q[0]) 
        n = 0
        while n < 2*N-4:
            X[n+2] = q[n+3] 
            X[n+3] = -L*q[n+3] - mt.cos(DSomeRotMode_t[ int(n/2 + 1) ])*q[n+2] + K*mt.cos( DSomeRotMode_t[ int(n/2 + 2) ]-DSomeRotMode_t[ int(n/2+1) ] )*(q[n+4] - q[n+2])   + K*mt.cos( DSomeRotMode_t[ int(n/2) ]-DSomeRotMode_t[ int(n/2+1) ] )*(q[n] - q[n+2])
            n+=2
        X[2*N-2] = q[2*N-1]
        X[2*N-1] = -L*q[2*N-1] - mt.cos(DSomeRotMode_t[N-1])*q[2*N-2] + K*mt.cos( DSomeRotMode_t[ N-2 ] - DSomeRotMode_t[ N-1 ] )*( q[2*N-4] - q[2*N-2] )
        return X
        
