import numpy as np

# Parameters
Rx_num = 1
Tx_num = 32
SC_num = 72


def DownPrecoding(channel_est):  ### Optional
    ### estimated channel
    channel_est = channel_est-0.5
    channel_est = np.reshape(channel_est, (-1, Rx_num, Tx_num, SC_num, 2)) ## Rx, Tx, Subcarrier, RealImag
    HH_complex_est = channel_est[:, :, :, :, 0] + 1j * channel_est[:, :, :, :, 1]  ## Rx, Tx, Subcarrier
    HH_complex_est = np.transpose(HH_complex_est, [0, 3, 1, 2])

    ### precoding based on the estimated channel
    MatRx, MatDiag, MatTx = np.linalg.svd(HH_complex_est, full_matrices=True) ## SVD
    PrecodingVector = np.conj(MatTx[:,:,0,:])  ## The best eigenvector (MRT transmission)
    PrecodingVector = np.reshape(PrecodingVector,(-1, SC_num, Tx_num, 1))
    return PrecodingVector


def EqChannelGain(channel, PrecodingVector):
    ### The authentic CSI
    channel = channel-0.5
    HH = np.reshape(channel, (-1, Rx_num, Tx_num, SC_num, 2))  ## Rx, Tx, Subcarrier, RealImag
    HH_complex = HH[:, :, :, :, 0] + 1j * HH[:, :, :, :, 1]  ## Rx, Tx, Subcarrier
    HH_complex = np.transpose(HH_complex, [0, 3, 1, 2])

    ### Power Normalization of the precoding vector
    Power = np.matmul(np.transpose(np.conj(PrecodingVector), (0, 1, 3, 2)), PrecodingVector)
    PrecodingVector = PrecodingVector / np.sqrt(Power)

    ### Effective channel gain
    R = np.matmul(HH_complex, PrecodingVector)
    R_conj = np.transpose(np.conj(R), (0, 1, 3, 2))
    h_sub_gain = np.matmul(R_conj, R)
    h_sub_gain = np.reshape(np.absolute(h_sub_gain), (-1, SC_num))  ### channel gain of SC_num subcarriers
    return h_sub_gain


def DataRate(h_sub_gain, sigma2_UE):  ### Score
    SNR = h_sub_gain / sigma2_UE
    Rate = np.log2(1 + SNR)  ## rate
    Rate_OFDM = np.mean(Rate, axis=-1)  ###  averaging over subcarriers
    Rate_OFDM_mean = np.mean(Rate_OFDM)  ### averaging over CSI samples
    return Rate_OFDM_mean