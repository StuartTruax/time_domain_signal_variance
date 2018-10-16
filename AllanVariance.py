import numpy as np
import warnings
class AllanVariance:
    @staticmethod
    def overlapped_variable_tau(X,Fs,*args):
        """
        Allan variance estimator using the overlapped variable tau method.
        
        Parameters
        ----------
        X: nparray
            Time series input data. All time steps are assumed to be uniformly spaced with sampling frequency Fs.
        Fs: float
            Sampling frequency of time series data.
        args[0]: float
             Minimum tau value for which the Allan variance is computed. Default is tau_0 = 1/Fs
        args[1]: float
            Maximum tau value for which the Allan variance is computed. Default is tau_0*(10^4)
        args[2]: int
            Number of logrithmically spaced points to be evaluated between the minimum and maximum
            tau values.
            
        Returns
        -------
        (taus, sigma_output): tuple
            tuple of output values
            
        taus: nparray
            tau values for which the Allan variance is computed.
            
        sigma_output: nparray
            Allan variance estimation for each of the tau values.
        
        Notes
        -----
        1. By subsectioning the input array in memory through range indexing, instead of summing over all  the indivdual elements and producing a scalar, the algorithm runs in close to linear time.
        2. np.mean captures the above summation and the 1/(N-2M-1) coefficient
        
        """
        warnings.filterwarnings("ignore")

        X = np.asarray(X, dtype=float)
        N = X.shape[0]
                
        tau_0 = 1/float(Fs)
            
        #range of cluster sizes
        M = []
        minLogM = 0
        maxLogM = 4
        numPoints = 40
        
        
        if args:
            if len(args) >= 1:
                minLogM = np.log(np.divide(args[0],tau_0))
            if len(args) >= 2:
                maxLogM = np.log(np.divide(args[1],tau_0))
            if len(args) <= 3:
                numPoints = args[2]
    
        if(minLogM < 0):
            minLogM = 0
        if( maxLogM > np.log(N)):
            maxLogM = np.floor(np.log(N))
        
        M = np.logspace(minLogM,maxLogM,numPoints,True).astype(int)
        
        taus = tau_0*M
        
        sigma_output = np.zeros(len(M))


        # key
        X = np.cumsum(X,axis=0)
        for i in range(0,len(M)):
            
            sum1 = X[2*M[i]:] - 2 * X[M[i]:-M[i]] + X[:-2*M[i]]
            sum1 = np.mean(sum1**2,axis=0)/ M[i]/ M[i]
            sigma_output[i] = np.divide(0.5*sum1,(tau_0)**2)

        
        #trim tau and sigma arrays to remove nans
        taus = taus[~np.isnan(sigma_output)]
        sigma_output = sigma_output [~np.isnan(sigma_output)]
        

        return taus,sigma_output


    @staticmethod
    def overlapped_variable_tau_rate(X,Fs,*tauLims):
        pass












        




