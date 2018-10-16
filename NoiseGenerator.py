import numpy as np
class NoiseGenerator:
    @staticmethod
    def power_law_noise(N,alpha,A_rms):
        """
            Generates a time-domain vector of power law noise samples.
            
            Parameters
            ----------
            N: int
                Number of samples to be generated from the power-law noise process.
            alpha: int
                Exponent paramter in the power-law noise process S(f)~h*f(-alpha).
            A_rms: float
                Root-mean squared (RMS) ampltiude of the generated time series
            
            Returns
            -------
            Y: nparray
                N-sample vector of samples from the specified noise process.
            
            Notes
            -----
        """
        
        N_nominal = N
        
        if(N%2 !=0):
            N_nominal = N+1
        
        halfSpectrumLength = N_nominal/2

        X_t = np.random.normal(0, 1, (N_nominal,1))
        X_omega = np.fft.fft(X_t)
        
        #shift the FFT to put the 0-frequency in the center
        X_omega = np.fft.fftshift(X_omega)
        
        #initialize colored noise spectrum
        X_omega_c = np.empty(N_nominal,np.dtype('c16'))
        
        
        #color the noise according to the exponent parameter
        #leave the DC component at X_omega[halfSpectrumLength] untouched
        for i in range(halfSpectrumLength+1,N_nominal):
            X_omega_c[i] = np.divide(X_omega[i],np.sqrt((i-halfSpectrumLength)**alpha))
        for i in range(halfSpectrumLength-1,-1,-1):
            X_omega_c[i] = np.divide(X_omega[i],np.sqrt((halfSpectrumLength-i)**alpha))



        #transform from frequency space back into real space
        Y = np.fft.ifft(np.fft.ifftshift(X_omega_c))
        Y = np.real(Y)
        

        #guarantee A_rms standard deviation and zero mean
        Y = Y - np.mean(Y)
        Y_rms = np.sqrt(np.mean(Y**2))
        Y = A_rms*np.divide(Y,Y_rms)

        return Y
