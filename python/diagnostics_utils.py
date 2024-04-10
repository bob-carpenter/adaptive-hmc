import numpy as np

def cumulative_mean(samples, mode=1):
    
    if len(samples.shape) == 1:
        return np.cumsum(samples**mode) / (1+np.arange(samples.size))
     
    if len(samples.shape) == 2: #assume shape (nchains, nsamples)
        return np.cumsum((samples.T)**mode) / (1+np.arange(samples.size))

    else:
        print("Not implemented for sample of shape : ", samples.shape)
        print("Expected 1-d or 2-d samples of structure (nchains, nsamples)")
            

# def cumulative_mean2(samples, mode=1):
    
#     if len(samples.shape) == 1:
#         return np.cumsum(samples**mode) / (1+np.arange(samples.size))

#     if len(samples.shape) == 2: #assume shape (nchains, nsamples)
#         cum_mean = np.array([(samples[:, :j]**mode).mean() for j in range(samples.shape[1])])
#         return cum_mean

#     else:
#         print("Not implemented for sample of shape : ", samples.shape)
#         print("Expected 1-d or 2-d samples of structure (nchains, nsamples)")
            

def cumulative_error(samples, counts, true_val=None, true_scatter=None, ref_samples=None, mode=1, relative=False, verbose=False):

    if (true_val is None) & (ref_samples is None):
        print("baseline is not given")
        raise
    if (true_scatter is None) & (ref_samples is None) & (relative == 'scatter'):
        print("Need reference samples or true scatter to normalize relative to scatter")
        raise

    if ref_samples is not None:
        assert len(ref_samples.shape) == 1
        true_val = np.mean(ref_samples**mode, axis=0)
        true_scatter = np.mean(ref_samples**mode, axis=0)
        if verbose: print(true_val)

    err = cumulative_mean(samples, mode) - true_val
        
    if relative=='val':
        err /= true_val
    elif relative=='scatter':
        err /= true_scatter
        
    count = np.cumsum(counts.T)
    return count, err

    

def cumulative_rmse(samples, counts, true_val=None, true_scatter=None, ref_samples=None, mode=1, relative=False, verbose=False):

    D = samples.shape[-1]
    if (true_val is None) & (ref_samples is None):
        print("baseline is not given")
        raise
    if (true_scatter is None) & (ref_samples is None) & (relative == 'scatter'):
        print("Need reference samples or true scatter to normalize relative to scatter")
        raise

    if ref_samples is not None:
        assert ref_samples.shape[-1] == D
        ref_samples = ref_samples.reshape(-1, D)
        true_val = np.mean(ref_samples**mode, axis=0)
        true_scatter = np.std(ref_samples**mode, axis=0)
        if verbose: print(true_val)

    errors = np.zeros((samples[..., 0].size, D))
    for d in range(D):
        err = cumulative_mean(samples[...,d], mode) - true_val[d]
        errors[:, d] = err

    if relative=='val':
        errors /= true_val
    elif relative=='scatter':
        errors /= true_scatter

    rmse = ((errors**2).mean(axis=-1))**0.5
    count = np.cumsum(counts.T)
    return count, rmse


# def cumulative_rmse2(samples, counts, true_val=None, true_scatter=None, ref_samples=None, mode=1, relative=False, verbose=False):

#     D = samples.shape[-1]
#     if (true_val is None) & (ref_samples is None):
#         print("baseline is not given")
#         raise
#     if (true_scatter is None) & (ref_samples is None) & (relative == 'scatter'):
#         print("Need reference samples or true scatter to normalize relative to scatter")
#         raise

#     if ref_samples is not None:
#         assert ref_samples.shape[-1] == D
#         ref_samples = ref_samples.reshape(-1, D)
#         true_val = np.mean(ref_samples**mode, axis=0)
#         true_scatter = np.std(ref_samples**mode, axis=0)
#         if verbose: print(true_val)

#     errors = np.zeros((samples.shape[1], D))
#     for d in range(D):
#         err = cumulative_mean2(samples[...,d], mode) - true_val[d]
#         errors[:, d] = err

#     if relative=='val':
#         errors /= true_val
#     elif relative=='scatter':
#         errors /= true_scatter

#     rmse = ((errors**2).mean(axis=-1))**0.5
#     count = np.array([(counts[:, :j]).sum() for j in range(samples.shape[1])])
#     return count, rmse



def cumulative_error_bootstrap(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
    assert len(samples.shape) == 2
    assert len(counts.shape) == 2
    nchains, nsamples = samples.shape

    count_list, err_list = [], []
    for i in range(nchains):
        idx = list(np.arange(nchains))
        idx.pop(i)
        count, err = cumulative_error(samples[idx], counts[idx], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
        count_list.append(count)
        err_list.append(err)

    return np.array(count_list), np.array(err_list)


def cumulative_error_scatter(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
    assert len(samples.shape) == 2
    assert len(counts.shape) == 2
    nchains, nsamples = samples.shape

    count_list, err_list = [], []
    for i in range(nchains):
        count, err = cumulative_error(samples[i:i+1], counts[i:i+1], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
        count_list.append(count)
        err_list.append(err)

    return np.array(count_list), np.array(err_list)


def cumulative_rmse_bootstrap(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
    assert len(samples.shape) == 3
    assert len(counts.shape) == 2
    nchains, nsamples, D = samples.shape

    count_list, err_list = [], []
    for i in range(nchains):
        idx = list(np.arange(nchains))
        idx.pop(i)
        count, err = cumulative_rmse(samples[idx], counts[idx], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
        count_list.append(count)
        err_list.append(err)

    return np.array(count_list), np.array(err_list)

                    

def cumulative_rmse_scatter(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
    assert len(samples.shape) == 3
    assert len(counts.shape) == 2
    nchains, nsamples, D = samples.shape

    count_list, err_list = [], []
    for i in range(nchains):
        count, err = cumulative_rmse(samples[i:i+1], counts[i:i+1], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
        count_list.append(count)
        err_list.append(err)

    return np.array(count_list), np.array(err_list)


# def cumulative_rmse_scatter2(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
#     assert len(samples.shape) == 3
#     assert len(counts.shape) == 2
#     nchains, nsamples, D = samples.shape

#     count_list, err_list = [], []
#     for i in range(nchains):
#         count, err = cumulative_rmse2(samples[i:i+1], counts[i:i+1], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
#         count_list.append(count)
#         err_list.append(err)

#     return np.array(count_list), np.array(err_list)
                    
